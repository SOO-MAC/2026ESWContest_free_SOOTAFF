#!/usr/bin/env python3
"""Build a reusable table-coordinate database from a completed OccupancyGrid.

Pipeline
--------
1. Receive the latest ``/map`` OccupancyGrid from SLAM Toolbox.
2. Downsample occupied cells and cluster them with DBSCAN.
3. Fit an oriented rectangle to every spatial cluster.
4. Keep only rectangle sizes that repeat at least ``min_repeated_count`` times.
5. For each rectangle, select the short edge facing the mapping start pose.
6. Place the table approach pose ``approach_offset_m`` outside that edge.
7. Number tables from the venue front toward the back, then left-to-right.
8. Save YAML and publish JSON + RViz markers.

Run the build only after mapping is sufficiently complete:
    ros2 service call /tables/build std_srvs/srv/Trigger "{}"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sklearn.cluster import DBSCAN
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
import yaml


@dataclass
class RectangleCandidate:
    center: np.ndarray
    corners: np.ndarray
    short_m: float
    long_m: float
    rect_yaw: float
    point_count: int


@dataclass
class TableRecord:
    table_id: int
    center_x: float
    center_y: float
    short_m: float
    long_m: float
    rect_yaw: float
    approach_x: float
    approach_y: float
    approach_yaw: float
    approach_offset_m: float
    front_distance_m: float
    lateral_distance_m: float
    corners: list[list[float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            'id': self.table_id,
            'center': {
                'x': round(self.center_x, 4),
                'y': round(self.center_y, 4),
            },
            'rectangle': {
                'short_m': round(self.short_m, 4),
                'long_m': round(self.long_m, 4),
                'yaw': round(self.rect_yaw, 6),
                'corners': [
                    {'x': round(x, 4), 'y': round(y, 4)}
                    for x, y in self.corners
                ],
            },
            'approach': {
                'x': round(self.approach_x, 4),
                'y': round(self.approach_y, 4),
                'yaw': round(self.approach_yaw, 6),
                'offset_m': round(self.approach_offset_m, 4),
            },
            'ordering': {
                'front_distance_m': round(self.front_distance_m, 4),
                'lateral_distance_m': round(self.lateral_distance_m, 4),
            },
        }


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class TableMapper(Node):
    def __init__(self) -> None:
        super().__init__('table_mapper')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('output_yaml', '~/.ros/venue_tables.yaml')
        self.declare_parameter('tables_topic', '/tables/data')
        self.declare_parameter('markers_topic', '/tables/markers')
        self.declare_parameter('build_service', '/tables/build')

        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('voxel_size_m', 0.08)
        self.declare_parameter('dbscan_eps_m', 0.34)
        self.declare_parameter('dbscan_min_samples', 4)
        self.declare_parameter('min_cluster_points', 18)

        # Bounding-size filter for one table+chairs group.
        self.declare_parameter('min_short_m', 0.35)
        self.declare_parameter('max_short_m', 1.60)
        self.declare_parameter('min_long_m', 0.55)
        self.declare_parameter('max_long_m', 2.40)
        self.declare_parameter('min_rect_area_m2', 0.25)
        self.declare_parameter('max_rect_area_m2', 3.80)

        # A second DBSCAN runs in (short, long) size space.  Only repeating
        # rectangle sizes are retained.
        self.declare_parameter('size_dbscan_eps_m', 0.20)
        self.declare_parameter('min_repeated_count', 3)

        self.declare_parameter('approach_offset_m', 0.30)
        self.declare_parameter('approach_free_radius_m', 0.22)
        self.declare_parameter('row_tolerance_m', 0.70)
        self.declare_parameter('number_left_to_right', True)

        # If all three override parameters are finite, they replace the first
        # map->base_link pose captured at mapping start.
        self.declare_parameter('use_start_override', False)
        self.declare_parameter('start_x_override', 0.0)
        self.declare_parameter('start_y_override', 0.0)
        self.declare_parameter('start_yaw_deg_override', 0.0)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.latest_map: Optional[OccupancyGrid] = None
        self.start_pose: Optional[tuple[float, float, float]] = None
        self.last_payload: Optional[dict[str, Any]] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('map_topic').value),
            self.map_callback,
            map_qos,
        )
        self.tables_publisher = self.create_publisher(
            String,
            str(self.get_parameter('tables_topic').value),
            latched_qos,
        )
        self.markers_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter('markers_topic').value),
            latched_qos,
        )
        self.create_service(
            Trigger,
            str(self.get_parameter('build_service').value),
            self.build_callback,
        )
        self.create_timer(0.20, self.capture_start_pose_once)

        self.get_logger().info(
            'Table mapper ready. Finish mapping, then call /tables/build.'
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def capture_start_pose_once(self) -> None:
        if self.start_pose is not None:
            return

        override = self.start_pose_override()
        if override is not None:
            self.start_pose = override
            self.get_logger().info(
                'Using configured venue start pose: '
                f'x={override[0]:.3f}, y={override[1]:.3f}, '
                f'yaw={math.degrees(override[2]):.1f} deg'
            )
            return

        map_frame = str(self.get_parameter('map_frame').value)
        base_frame = str(self.get_parameter('base_frame').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                Time(),
            )
        except TransformException:
            return

        q = transform.transform.rotation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.start_pose = (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            yaw,
        )
        self.get_logger().info(
            'Captured mapping-start pose: '
            f'x={self.start_pose[0]:.3f}, y={self.start_pose[1]:.3f}, '
            f'yaw={math.degrees(self.start_pose[2]):.1f} deg'
        )

    def start_pose_override(self) -> Optional[tuple[float, float, float]]:
        if not bool(self.get_parameter('use_start_override').value):
            return None
        x = float(self.get_parameter('start_x_override').value)
        y = float(self.get_parameter('start_y_override').value)
        yaw_deg = float(self.get_parameter('start_yaw_deg_override').value)
        return x, y, math.radians(yaw_deg)

    def build_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        if self.latest_map is None:
            response.success = False
            response.message = '아직 /map을 받지 못했습니다.'
            return response
        if self.start_pose is None:
            response.success = False
            response.message = '아직 mapping start pose를 잡지 못했습니다.'
            return response

        try:
            records = self.detect_tables(self.latest_map)
            if not records:
                response.success = False
                response.message = (
                    '반복되는 테이블 직사각형을 찾지 못했습니다. '
                    'RViz marker와 YAML 파라미터를 조정하세요.'
                )
                return response

            payload = self.make_payload(records, self.latest_map)
            payload = self.merge_manual_fields(payload)
            output_path = self.save_yaml(payload)
            self.publish_payload(payload)
            self.publish_markers(records)
            self.last_payload = payload

            response.success = True
            response.message = (
                f'{len(records)}개 테이블 저장 완료: {output_path}'
            )
            self.get_logger().info(response.message)
            return response
        except Exception as error:  # noqa: BLE001 - ROS service must respond.
            self.get_logger().exception(f'Table build failed: {error}')
            response.success = False
            response.message = f'테이블 분석 실패: {error}'
            return response

    def detect_tables(self, grid: OccupancyGrid) -> list[TableRecord]:
        points = self.occupied_world_points(grid)
        if len(points) < 10:
            self.get_logger().warning('Occupied points are too few.')
            return []

        voxel_points = self.voxel_downsample(points)
        spatial_labels = DBSCAN(
            eps=float(self.get_parameter('dbscan_eps_m').value),
            min_samples=int(self.get_parameter('dbscan_min_samples').value),
            algorithm='kd_tree',
            n_jobs=-1,
        ).fit_predict(voxel_points)

        candidates: list[RectangleCandidate] = []
        min_cluster_points = int(self.get_parameter('min_cluster_points').value)
        for label in sorted(set(int(value) for value in spatial_labels)):
            if label < 0:
                continue
            cluster = voxel_points[spatial_labels == label]
            if len(cluster) < min_cluster_points:
                continue
            candidate = self.fit_rectangle(cluster)
            if candidate is not None:
                candidates.append(candidate)

        self.get_logger().info(
            f'Spatial DBSCAN: {len(candidates)} rectangle candidates'
        )
        repeated = self.keep_repeated_sizes(candidates)
        self.get_logger().info(
            f'Repeated-size filter: {len(repeated)} candidates retained'
        )

        raw_records: list[TableRecord] = []
        for candidate in repeated:
            record = self.candidate_to_record(candidate, grid)
            if record is not None:
                raw_records.append(record)

        return self.assign_table_numbers(raw_records)

    def occupied_world_points(self, grid: OccupancyGrid) -> np.ndarray:
        width = int(grid.info.width)
        height = int(grid.info.height)
        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        threshold = int(self.get_parameter('occupied_threshold').value)
        rows, cols = np.where(data >= threshold)
        if len(rows) == 0:
            return np.empty((0, 2), dtype=np.float64)

        resolution = float(grid.info.resolution)
        local_x = (cols.astype(np.float64) + 0.5) * resolution
        local_y = (rows.astype(np.float64) + 0.5) * resolution

        origin = grid.info.origin
        origin_yaw = quaternion_to_yaw(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )
        c = math.cos(origin_yaw)
        s = math.sin(origin_yaw)
        world_x = float(origin.position.x) + c * local_x - s * local_y
        world_y = float(origin.position.y) + s * local_x + c * local_y
        return np.column_stack((world_x, world_y))

    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        voxel = max(float(self.get_parameter('voxel_size_m').value), 1e-3)
        keys = np.floor(points / voxel).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(unique_indices)]

    def fit_rectangle(self, cluster: np.ndarray) -> Optional[RectangleCandidate]:
        rect = cv2.minAreaRect(cluster.astype(np.float32).reshape((-1, 1, 2)))
        corners = cv2.boxPoints(rect).astype(np.float64)
        edge_vectors = np.roll(corners, -1, axis=0) - corners
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        short_m = float(np.min(edge_lengths))
        long_m = float(np.max(edge_lengths))
        area = short_m * long_m

        if not (
            float(self.get_parameter('min_short_m').value)
            <= short_m
            <= float(self.get_parameter('max_short_m').value)
        ):
            return None
        if not (
            float(self.get_parameter('min_long_m').value)
            <= long_m
            <= float(self.get_parameter('max_long_m').value)
        ):
            return None
        if not (
            float(self.get_parameter('min_rect_area_m2').value)
            <= area
            <= float(self.get_parameter('max_rect_area_m2').value)
        ):
            return None

        long_index = int(np.argmax(edge_lengths))
        long_vector = edge_vectors[long_index]
        rect_yaw = math.atan2(float(long_vector[1]), float(long_vector[0]))
        center = np.asarray(rect[0], dtype=np.float64)
        return RectangleCandidate(
            center=center,
            corners=corners,
            short_m=short_m,
            long_m=long_m,
            rect_yaw=normalize_angle(rect_yaw),
            point_count=len(cluster),
        )

    def keep_repeated_sizes(
        self,
        candidates: list[RectangleCandidate],
    ) -> list[RectangleCandidate]:
        min_count = int(self.get_parameter('min_repeated_count').value)
        if len(candidates) < min_count:
            return []

        sizes = np.asarray(
            [[item.short_m, item.long_m] for item in candidates],
            dtype=np.float64,
        )
        labels = DBSCAN(
            eps=float(self.get_parameter('size_dbscan_eps_m').value),
            min_samples=min_count,
        ).fit_predict(sizes)
        return [
            item
            for item, label in zip(candidates, labels, strict=True)
            if int(label) >= 0
        ]

    def candidate_to_record(
        self,
        candidate: RectangleCandidate,
        grid: OccupancyGrid,
    ) -> Optional[TableRecord]:

        assert self.start_pose is not None

        start_xy = np.asarray(
            self.start_pose[:2],
            dtype=np.float64,
        )

        corners = candidate.corners

        edge_vectors = (
            np.roll(corners, -1, axis=0)
            - corners
        )

        edge_lengths = np.linalg.norm(
            edge_vectors,
            axis=1,
        )

        # 직사각형의 짧은 변 2개
        short_indices = np.argsort(
            edge_lengths
        )[:2]

        short_edge_centers = [
            (
                corners[index]
                + corners[(index + 1) % 4]
            ) / 2.0
            for index in short_indices
        ]

        # ----------------------------------------------------
        # 입구(start pose)에 가까운 짧은 변부터 검사
        # ----------------------------------------------------

        short_edge_centers.sort(
            key=lambda point: float(
                np.linalg.norm(
                    point - start_xy
                )
            )
        )

        offset = float(
            self.get_parameter(
                'approach_offset_m'
            ).value
        )

        free_radius = float(
            self.get_parameter(
                'approach_free_radius_m'
            ).value
        )

        selected_approach = None
        selected_edge_index = None

        # ----------------------------------------------------
        # 1순위: 입구쪽 짧은 변
        # 2순위: 반대쪽 짧은 변
        # ----------------------------------------------------

        for edge_number, edge_center in enumerate(
            short_edge_centers
        ):

            outward = (
                edge_center
                - candidate.center
            )

            norm = float(
                np.linalg.norm(outward)
            )

            if norm < 1e-6:
                continue

            outward = outward / norm

            approach = (
                edge_center
                + outward * offset
            )

            if self.is_free_circle(
                grid,
                approach,
                free_radius,
            ):

                selected_approach = approach
                selected_edge_index = edge_number

                break

        # ----------------------------------------------------
        # 양쪽 모두 접근 불가능한 경우에만 제외
        # ----------------------------------------------------

        if selected_approach is None:

            self.get_logger().warning(
                'Rejected table candidate: '
                'both short-edge approach points '
                f'are blocked. '
                f'center=({candidate.center[0]:.2f}, '
                f'{candidate.center[1]:.2f})'
            )

            return None

        approach = selected_approach

        # 입구쪽이 막혀 반대쪽을 사용한 경우
        if selected_edge_index == 1:

            self.get_logger().warning(
                'Entrance-side approach blocked; '
                'using opposite short edge: '
                f'x={approach[0]:.2f}, '
                f'y={approach[1]:.2f}'
            )

        # 로봇은 테이블 중심을 바라본다.
        approach_yaw = math.atan2(
            float(
                candidate.center[1]
                - approach[1]
            ),
            float(
                candidate.center[0]
                - approach[0]
            ),
        )

        front, lateral = (
            self.venue_coordinates(
                candidate.center
            )
        )

        return TableRecord(
            table_id=0,

            center_x=float(
                candidate.center[0]
            ),

            center_y=float(
                candidate.center[1]
            ),

            short_m=candidate.short_m,
            long_m=candidate.long_m,
            rect_yaw=candidate.rect_yaw,

            approach_x=float(
                approach[0]
            ),

            approach_y=float(
                approach[1]
            ),

            approach_yaw=normalize_angle(
                approach_yaw
            ),

            approach_offset_m=offset,

            front_distance_m=front,
            lateral_distance_m=lateral,

            corners=[
                [float(x), float(y)]
                for x, y in corners
            ],
        )

    def venue_coordinates(self, point: np.ndarray) -> tuple[float, float]:
        assert self.start_pose is not None
        origin = np.asarray(self.start_pose[:2], dtype=np.float64)
        yaw = self.start_pose[2]
        delta = point - origin
        front_axis = np.asarray([math.cos(yaw), math.sin(yaw)])
        left_axis = np.asarray([-math.sin(yaw), math.cos(yaw)])
        return float(delta @ front_axis), float(delta @ left_axis)

    def assign_table_numbers(
        self,
        records: list[TableRecord],
    ) -> list[TableRecord]:
        if not records:
            return []

        row_tolerance = float(self.get_parameter('row_tolerance_m').value)
        sorted_front = sorted(records, key=lambda item: item.front_distance_m)
        rows: list[list[TableRecord]] = []
        row_means: list[float] = []

        for record in sorted_front:
            if not rows or abs(record.front_distance_m - row_means[-1]) > row_tolerance:
                rows.append([record])
                row_means.append(record.front_distance_m)
            else:
                rows[-1].append(record)
                row_means[-1] = sum(
                    item.front_distance_m for item in rows[-1]
                ) / len(rows[-1])

        left_to_right = bool(self.get_parameter('number_left_to_right').value)
        ordered: list[TableRecord] = []
        for row in rows:
            row.sort(
                key=lambda item: item.lateral_distance_m,
                reverse=left_to_right,
            )
            ordered.extend(row)

        for index, record in enumerate(ordered, start=1):
            record.table_id = index
        return ordered

    def world_to_grid(
        self,
        grid: OccupancyGrid,
        point: np.ndarray,
    ) -> tuple[int, int]:
        origin = grid.info.origin
        yaw = quaternion_to_yaw(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )
        dx = float(point[0] - origin.position.x)
        dy = float(point[1] - origin.position.y)
        c = math.cos(yaw)
        s = math.sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        resolution = float(grid.info.resolution)
        return int(math.floor(local_x / resolution)), int(math.floor(local_y / resolution))

    def is_free_circle(
        self,
        grid: OccupancyGrid,
        point: np.ndarray,
        radius_m: float,
    ) -> bool:
        width = int(grid.info.width)
        height = int(grid.info.height)
        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        col, row = self.world_to_grid(grid, point)
        resolution = float(grid.info.resolution)
        radius_cells = max(1, int(math.ceil(radius_m / resolution)))
        threshold = int(self.get_parameter('occupied_threshold').value)

        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if d_row * d_row + d_col * d_col > radius_cells * radius_cells:
                    continue
                test_row = row + d_row
                test_col = col + d_col
                if not (0 <= test_row < height and 0 <= test_col < width):
                    return False
                value = int(data[test_row, test_col])
                if value < 0 or value >= threshold:
                    return False
        return True

    def make_payload(
        self,
        records: list[TableRecord],
        grid: OccupancyGrid,
    ) -> dict[str, Any]:
        assert self.start_pose is not None
        start_x, start_y, start_yaw = self.start_pose
        return {
            'version': 1,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'frame_id': grid.header.frame_id or str(self.get_parameter('map_frame').value),
            'pickup_zone': {
                'x': round(start_x, 4),
                'y': round(start_y, 4),
                'yaw': round(start_yaw, 6),
            },
            'numbering': {
                'origin_x': round(start_x, 4),
                'origin_y': round(start_y, 4),
                'front_yaw': round(start_yaw, 6),
                'rule': 'front_to_back_then_left_to_right',
                'row_tolerance_m': float(
                    self.get_parameter('row_tolerance_m').value
                ),
            },
            'table_count': len(records),
            'tables': [record.as_dict() for record in records],
        }

    def merge_manual_fields(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Preserve manually registered venue information.

        /tables/build regenerates automatic table information, but
        seat_to_table and named_locations are manually configured.
        If venue_tables.yaml already contains those fields, keep them.
        """

        output = Path(
            str(
                self.get_parameter(
                    'output_yaml'
                ).value
            )
        ).expanduser()

        if not output.is_file():
            return payload

        try:
            with output.open(
                'r',
                encoding='utf-8',
            ) as stream:
                existing = yaml.safe_load(stream)

        except Exception as error:
            self.get_logger().warning(
                '기존 venue YAML을 읽지 못해 '
                'manual field를 보존하지 못했습니다: '
                f'{error}'
            )
            return payload

        if not isinstance(existing, dict):
            return payload

        preserved = []

        for key in (
            'seat_to_table',
            'named_locations',
        ):
            if key in existing:
                payload[key] = existing[key]
                preserved.append(key)

        if preserved:
            self.get_logger().info(
                'Preserved manual venue fields: '
                + ', '.join(preserved)
            )

        return payload

    def save_yaml(self, payload: dict[str, Any]) -> Path:
        output = Path(
            str(self.get_parameter('output_yaml').value)
        ).expanduser()

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Preserve logical seat-to-table mapping.
        #
        # /tables/build는 물리적인 테이블 좌표를 다시 생성하지만
        # seat_to_table은 행사 운영용 논리 정보이므로 기존 값을
        # 유지한다.
        #
        # 새 맵에 존재하지 않는 table_id를 가리키는 좌석은
        # 안전을 위해 제거한다.
        # --------------------------------------------------------

        preserved_seat_to_table = {}

        if output.is_file():

            try:

                previous = yaml.safe_load(
                    output.read_text(
                        encoding='utf-8'
                    )
                ) or {}

                previous_mapping = previous.get(
                    'seat_to_table',
                    {},
                )

                valid_table_ids = {
                    int(table['id'])
                    for table in payload.get(
                        'tables',
                        [],
                    )
                    if isinstance(table, dict)
                    and 'id' in table
                }

                if isinstance(
                    previous_mapping,
                    dict,
                ):

                    for seat_id, table_id in (
                        previous_mapping.items()
                    ):

                        try:

                            normalized_seat_id = int(
                                seat_id
                            )

                            normalized_table_id = int(
                                table_id
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):

                            continue

                        if (
                            normalized_table_id
                            in valid_table_ids
                        ):

                            preserved_seat_to_table[
                                normalized_seat_id
                            ] = normalized_table_id

                if preserved_seat_to_table:

                    self.get_logger().info(
                        'Preserved '
                        f'{len(preserved_seat_to_table)} '
                        'seat_to_table entries.'
                    )

            except Exception as error:  # noqa: BLE001

                self.get_logger().warning(
                    '기존 seat_to_table을 '
                    f'읽지 못했습니다: {error}'
                )

        payload['seat_to_table'] = (
            preserved_seat_to_table
        )

        # named location은 이전 맵의 좌표를 그대로 사용하면
        # 위험할 수 있으므로 보존하지 않는다.
        payload.setdefault(
            'named_locations',
            {},
        )

        temporary = output.with_suffix(
            output.suffix + '.tmp'
        )

        with temporary.open(
            'w',
            encoding='utf-8',
        ) as stream:

            yaml.safe_dump(
                payload,
                stream,
                allow_unicode=True,
                sort_keys=False,
            )

        temporary.replace(output)

        return output

    def publish_payload(self, payload: dict[str, Any]) -> None:
        self.tables_publisher.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

    def publish_markers(self, records: list[TableRecord]) -> None:
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        frame_id = str(self.get_parameter('map_frame').value)
        now = self.get_clock().now().to_msg()
        for record in records:
            base_id = record.table_id * 10

            rectangle = Marker()
            rectangle.header.frame_id = frame_id
            rectangle.header.stamp = now
            rectangle.ns = 'table_rectangles'
            rectangle.id = base_id
            rectangle.type = Marker.CUBE
            rectangle.action = Marker.ADD
            rectangle.pose.position.x = record.center_x
            rectangle.pose.position.y = record.center_y
            rectangle.pose.position.z = 0.04
            rectangle.pose.orientation.z = math.sin(record.rect_yaw / 2.0)
            rectangle.pose.orientation.w = math.cos(record.rect_yaw / 2.0)
            rectangle.scale.x = record.long_m
            rectangle.scale.y = record.short_m
            rectangle.scale.z = 0.08
            rectangle.color.r = 0.10
            rectangle.color.g = 0.70
            rectangle.color.b = 0.90
            rectangle.color.a = 0.35
            marker_array.markers.append(rectangle)

            approach = Marker()
            approach.header.frame_id = frame_id
            approach.header.stamp = now
            approach.ns = 'table_approach'
            approach.id = base_id + 1
            approach.type = Marker.ARROW
            approach.action = Marker.ADD
            approach.pose.position.x = record.approach_x
            approach.pose.position.y = record.approach_y
            approach.pose.position.z = 0.08
            approach.pose.orientation.z = math.sin(record.approach_yaw / 2.0)
            approach.pose.orientation.w = math.cos(record.approach_yaw / 2.0)
            approach.scale.x = 0.35
            approach.scale.y = 0.09
            approach.scale.z = 0.09
            approach.color.r = 0.15
            approach.color.g = 0.95
            approach.color.b = 0.20
            approach.color.a = 0.95
            marker_array.markers.append(approach)

            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = now
            label.ns = 'table_labels'
            label.id = base_id + 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = record.center_x
            label.pose.position.y = record.center_y
            label.pose.position.z = 0.45
            label.pose.orientation.w = 1.0
            label.scale.z = 0.34
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = str(record.table_id)
            marker_array.markers.append(label)

            line = Marker()
            line.header.frame_id = frame_id
            line.header.stamp = now
            line.ns = 'table_offset_lines'
            line.id = base_id + 3
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.03
            line.color.r = 1.0
            line.color.g = 0.75
            line.color.b = 0.10
            line.color.a = 0.95
            line.points = [
                Point(x=record.approach_x, y=record.approach_y, z=0.06),
                Point(x=record.center_x, y=record.center_y, z=0.06),
            ]
            marker_array.markers.append(line)

        self.markers_publisher.publish(marker_array)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = TableMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
