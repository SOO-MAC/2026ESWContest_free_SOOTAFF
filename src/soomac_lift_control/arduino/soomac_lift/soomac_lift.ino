// =====================================
// SOOMAC LIFT ARDUINO FINAL CODE
// Linear Actuator 2ea + IBT-2 2ea
// USB Serial Control
// =====================================

// ===============================
// IBT-2 #1 pin
// ===============================
const int M1_RPWM = 5;
const int M1_LPWM = 6;
const int M1_R_EN = 7;
const int M1_L_EN = 8;

// ===============================
// IBT-2 #2 pin
// ===============================
const int M2_RPWM = 9;
const int M2_LPWM = 10;
const int M2_R_EN = 11;
const int M2_L_EN = 12;

// ===============================
// Speed tuning
// 범위: 0 ~ 255
// 빠른 모터 쪽 값을 낮추면 됨.
// ===============================
const int M1_UP_SPEED = 244.5;
const int M1_DOWN_SPEED = 255;

const int M2_UP_SPEED = 255;
const int M2_DOWN_SPEED = 245;

// ROS2가 7초 뒤 STOP을 보내지만,
// 혹시 STOP이 안 와도 8초 뒤 자동 정지
const unsigned long SAFETY_STOP_TIME_MS = 12000;

String serialBuffer = "";

bool isMoving = false;
unsigned long moveStartTime = 0;


// ===============================
// Stop all motors
// ===============================
void stopLift() {
  analogWrite(M1_RPWM, 0);
  analogWrite(M1_LPWM, 0);

  analogWrite(M2_RPWM, 0);
  analogWrite(M2_LPWM, 0);

  isMoving = false;

  Serial.println("STOPPED");
}


// ===============================
// Motor 1 only test
// ===============================
void motor1Up() {
  analogWrite(M1_RPWM, 0);
  analogWrite(M1_LPWM, M1_UP_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("MOTOR 1 UP");
}

void motor1Down() {
  analogWrite(M1_LPWM, 0);
  analogWrite(M1_RPWM, M1_DOWN_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("MOTOR 1 DOWN");
}

void motor2Up() {
  analogWrite(M2_RPWM, 0);
  analogWrite(M2_LPWM, M2_UP_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("MOTOR 2 UP");
}

void motor2Down() {
  analogWrite(M2_LPWM, 0);
  analogWrite(M2_RPWM, M2_DOWN_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("MOTOR 2 DOWN");
}


// ===============================
// Both motors UP
// ===============================
void liftUp() {
  analogWrite(M1_RPWM, 0);
  analogWrite(M1_LPWM, M1_UP_SPEED);

  analogWrite(M2_RPWM, 0);
  analogWrite(M2_LPWM, M2_UP_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("LIFT UP START - BOTH MOTORS");
}


void liftDown() {
  analogWrite(M1_LPWM, 0);
  analogWrite(M1_RPWM, M1_DOWN_SPEED);

  analogWrite(M2_LPWM, 0);
  analogWrite(M2_RPWM, M2_DOWN_SPEED);

  isMoving = true;
  moveStartTime = millis();

  Serial.println("LIFT DOWN START - BOTH MOTORS");
}


// ===============================
// Command handler
// ===============================
void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) {
    return;
  }

  Serial.print("RECEIVED: ");
  Serial.println(cmd);

  if (cmd == "UP") {
    liftUp();
  }
  else if (cmd == "DOWN") {
    liftDown();
  }
  else if (cmd == "STOP") {
    stopLift();
  }
  else if (cmd == "HOME") {
    stopLift();
    Serial.println("HOME COMMAND: STOP ONLY");
  }

  // 개별 모터 테스트용
  else if (cmd == "M1_UP") {
    motor1Up();
  }
  else if (cmd == "M1_DOWN") {
    motor1Down();
  }
  else if (cmd == "M2_UP") {
    motor2Up();
  }
  else if (cmd == "M2_DOWN") {
    motor2Down();
  }
  else {
    Serial.println("UNKNOWN COMMAND");
  }
}


// ===============================
// Serial read
// ===============================
void readSerialCommand() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(serialBuffer);
      serialBuffer = "";
    }
    else if (c != '\r') {
      serialBuffer += c;
    }
  }
}


// ===============================
// Safety auto stop
// ===============================
void updateSafetyStop() {
  if (!isMoving) {
    return;
  }

  unsigned long elapsed = millis() - moveStartTime;

  if (elapsed >= SAFETY_STOP_TIME_MS) {
    stopLift();
    Serial.println("SAFETY AUTO STOP");
  }
}


// ===============================
// Setup
// ===============================
void setup() {
  pinMode(M1_RPWM, OUTPUT);
  pinMode(M1_LPWM, OUTPUT);
  pinMode(M1_R_EN, OUTPUT);
  pinMode(M1_L_EN, OUTPUT);

  pinMode(M2_RPWM, OUTPUT);
  pinMode(M2_LPWM, OUTPUT);
  pinMode(M2_R_EN, OUTPUT);
  pinMode(M2_L_EN, OUTPUT);

  digitalWrite(M1_R_EN, HIGH);
  digitalWrite(M1_L_EN, HIGH);

  digitalWrite(M2_R_EN, HIGH);
  digitalWrite(M2_L_EN, HIGH);

  stopLift();

  Serial.begin(115200);
  Serial.println("SOOMAC USB LIFT READY - 2 MOTORS FINAL");
}


// ===============================
// Loop
// ===============================
void loop() {
  readSerialCommand();
  updateSafetyStop();
}
