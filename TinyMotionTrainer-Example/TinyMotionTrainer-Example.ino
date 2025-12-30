// Continuous sliding-window inference version with "stressed hold"
// and merged calm/still state
// WORKING VERSION WITH OLED OUTPUT (CALM / STRESSED GRAPHICS)

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <math.h>

#include <U8g2lib.h>
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

#include "model.h"

//====================================
// Settings
//====================================
#define MOTION_THRESHOLD    0.1    // gate inference when moving
#define NUM_SAMPLES         200    // window length (must match model)
#define INFERENCE_INTERVAL  10     // run model every 10 new samples

// Stressed hold behaviour
#define STRESSED_HOLD_MS    1500   // keep "Stressy" state for 1.5s after last strong hit
#define CONF_STRESSED       0.60f  // min conf to latch Stressy
#define CONF_OTHER          0.60f  // min conf to accept calm

// Model labels (raw)
const char *GESTURES[] = {"Stressy", "calmly", "still"};

// We will only expose two states externally:
// -1 = unknown, 0 = Stressy, 1 = calmly (calm + still)

//====================================
// OLED SETUP
//====================================

// 1.3" 128x32 OLED via I2C – same as your example
U8G2_SSD1306_128X32_UNIVISION_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

// Frame timing & pixel counts (copied from your OLED sketch)
const unsigned long CALM_FRAME_INTERVAL   = 150;   // ms between calm frames
const unsigned long STRESS_FRAME_INTERVAL = 40;    // ms between stressed frames

const uint8_t CALM_PIXEL_COUNT   = 20;   // few pixels on screen
const uint8_t STRESS_PIXEL_COUNT = 90;   // lots of pixels on screen

unsigned long lastOledFrameTime = 0;

// Prepare font / draw state
void u8g2_prepare() {
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.setFontRefHeightExtendedText();
  u8g2.setDrawColor(1);
  u8g2.setFontPosTop();
  u8g2.setFontDirection(0);
}

// Calm frame (same as your working example)
void drawCalmFrame() {
  u8g2.clearBuffer();
  u8g2_prepare();

  // Text at the top
  u8g2.drawStr(0, 0, "I am calm");

  // Draw a small number of random pixels, slow changes
  for (uint8_t i = 0; i < CALM_PIXEL_COUNT; i++) {
    uint8_t x = random(0, 128);
    uint8_t y = random(12, 32);  // keep below text
    u8g2.drawPixel(x, y);
  }

  u8g2.sendBuffer();
}

// Stressed frame (same as your working example)
void drawStressedFrame() {
  u8g2.clearBuffer();
  u8g2_prepare();

  // Text at the top
  u8g2.drawStr(0, 0, "I am stressed");

  // Draw lots of random pixels with clusters to feel chaotic
  for (uint8_t i = 0; i < STRESS_PIXEL_COUNT; i++) {
    uint8_t x = random(0, 128);
    uint8_t y = random(12, 32);
    u8g2.drawPixel(x, y);

    // occasionally draw a tiny cluster around that pixel
    if (random(0, 4) == 0) { // 1/4 chance to add neighbours
      int8_t dx = (int8_t)random(-1, 2); // -1,0,1
      int8_t dy = (int8_t)random(-1, 2);
      int16_t nx = x + dx;
      int16_t ny = y + dy;
      if (nx >= 0 && nx < 128 && ny >= 12 && ny < 32) {
        u8g2.drawPixel(nx, ny);
      }
    }
  }

  u8g2.sendBuffer();
}

// Called every loop to keep the OLED animated according to currentState
void updateOledForState(int state) {
  // Don't show anything until we have a known state
  if (state != 0 && state != 1) {
    return;
  }

  unsigned long now = millis();
  unsigned long frameInterval =
    (state == 0) ? STRESS_FRAME_INTERVAL : CALM_FRAME_INTERVAL;

  if (now - lastOledFrameTime < frameInterval) {
    return;
  }
  lastOledFrameTime = now;

  if (state == 0) {
    drawStressedFrame();
  } else {
    drawCalmFrame();
  }
}

//====================================
// TensorFlow / model globals
//====================================
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

//====================================
// Sliding window buffer
//====================================
float imuBuffer[NUM_SAMPLES * 6];
int bufferIndex = 0;          // where the NEXT sample will be written
bool bufferFilled = false;    // becomes true after first NUM_SAMPLES samples
int samplesSinceLastInference = 0;

//====================================
// Latched state for output
//====================================
// -1 = unknown, 0 = Stressy, 1 = calmly (merged calm+still)
int currentState = -1;
unsigned long lastStressedTime = 0;

//====================================
// Setup
//====================================
void setup() {
  Serial.begin(9600);
  // while (!Serial);  // OK for debugging. Comment out for battery-only use.

  // OLED init
  u8g2.begin();
  // Seed randomness for pixel positions
  randomSeed(analogRead(A0));

  if (!IMU.begin()) {
    Serial.println("IMU ERROR");
    while (1);
  }

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("MODEL VERSION MISMATCH");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(
      tflModel,
      tflOpsResolver,
      tensorArena,
      tensorArenaSize,
      &tflErrorReporter);

  tflInterpreter->AllocateTensors();
  tflInputTensor  = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("SLIDING WINDOW + STRESSED HOLD MODE (CALM/STILL MERGED)");
  Serial.println("States: Stressy / calmly");
  Serial.println();
}

//====================================
// Main loop: continuous streaming
//====================================
void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Only act when both accel and gyro have new data
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);

    // Normalize to match training
    float nAx = aX / 4.0;
    float nAy = aY / 4.0;
    float nAz = aZ / 4.0;
    float nGx = gX / 2000.0;
    float nGy = gY / 2000.0;
    float nGz = gZ / 2000.0;

    // Put into rolling buffer at current bufferIndex
    int base = bufferIndex * 6;
    imuBuffer[base + 0] = nAx;
    imuBuffer[base + 1] = nAy;
    imuBuffer[base + 2] = nAz;
    imuBuffer[base + 3] = nGx;
    imuBuffer[base + 4] = nGy;
    imuBuffer[base + 5] = nGz;

    // Advance circular index
    bufferIndex++;
    if (bufferIndex >= NUM_SAMPLES) {
      bufferIndex = 0;
      bufferFilled = true;  // after first full cycle
    }

    // Count samples since last inference
    samplesSinceLastInference++;

    // Compute current motion magnitude to gate inference
    float avgMag = (fabs(nAx) + fabs(nAy) + fabs(nAz)
                  + fabs(nGx) + fabs(nGy) + fabs(nGz)) / 6.0;

    // Only run inference if:
    // - buffer is full (we have at least NUM_SAMPLES)
    // - enough new samples since last inference
    // - we are above motion threshold (ignore fully still-ish)
    if (bufferFilled &&
        samplesSinceLastInference >= INFERENCE_INTERVAL &&
        avgMag >= MOTION_THRESHOLD) {

      samplesSinceLastInference = 0;  // reset interval counter

      // Copy sliding window into model input in time order:
      // oldest → newest, starting from bufferIndex (which points to "next write")
      for (int i = 0; i < NUM_SAMPLES; i++) {
        int srcIndex = (bufferIndex + i) % NUM_SAMPLES;  // circular read
        int srcBase  = srcIndex * 6;
        int dstBase  = i * 6;

        tflInputTensor->data.f[dstBase + 0] = imuBuffer[srcBase + 0];
        tflInputTensor->data.f[dstBase + 1] = imuBuffer[srcBase + 1];
        tflInputTensor->data.f[dstBase + 2] = imuBuffer[srcBase + 2];
        tflInputTensor->data.f[dstBase + 3] = imuBuffer[srcBase + 3];
        tflInputTensor->data.f[dstBase + 4] = imuBuffer[srcBase + 4];
        tflInputTensor->data.f[dstBase + 5] = imuBuffer[srcBase + 5];
      }

      // Run inference
      TfLiteStatus status = tflInterpreter->Invoke();
      if (status != kTfLiteOk) {
        Serial.println("INFERENCE ERROR");
        return;
      }

      // Find raw winner for this window
      int   maxIndex = 0;
      float maxValue = -1.0f;
      for (int i = 0; i < 3; i++) {
        float v = tflOutputTensor->data.f[i];
        if (v > maxValue) {
          maxValue = v;
          maxIndex = i;
        }
      }

      unsigned long now = millis();

      // 1) If we see a strong "Stressy" prediction, latch it and update timer
      if (maxIndex == 0 && maxValue >= CONF_STRESSED) {
        if (currentState != 0) {
          currentState = 0;
          Serial.print("STATE: Stressy (conf ");
          Serial.print(maxValue, 3);
          Serial.println(")");
        }
        lastStressedTime = now;
      } 
      else {
        // Not currently a strong stressed frame

        // 2) If we're in stressed state and the hold time hasn't expired, keep it
        if (currentState == 0 && (now - lastStressedTime) < STRESSED_HOLD_MS) {
          // stay in Stressy, do nothing
        } 
        else {
          // 3) Stressed hold expired OR we weren't stressed:
          //    Treat any confident calmly/still as one "calmly" state

          if (maxValue >= CONF_OTHER && (maxIndex == 1 || maxIndex == 2)) {
            int newState = 1;  // merged calm state

            if (newState != currentState) {
              currentState = newState;
              Serial.print("STATE: calmly (conf ");
              Serial.print(maxValue, 3);
              Serial.println(")");
            }
          }
          // if confidence is low, ignore and keep previous state
        }
      }
    }
  }

  // Update OLED animation based on currentState
  updateOledForState(currentState);
}






