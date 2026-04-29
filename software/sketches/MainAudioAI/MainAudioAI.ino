/**
 * LIGHTONY ROBOT - Main Audio AI Control
 * * This sketch implements the core voice recognition and lamp control logic 
 * for the LIGHTONY robot.
 * * Acknowledgments:
 * This code is developed with reference to the "MainAudioAI.ino" sample from:
 * - Book: "SPRESENSEではじめるローパワーエッジAI" by Yoshinori Oota
 * - Repository: https://github.com/TE-YoshinoriOota/Spresense-LowPower-EdgeAI
 * * The original logic for Spresense-based Edge AI inference has been 
 * adapted and integrated into the LIGHTONY project.
 */

#ifdef SUBCORE
#error "Core selection is wrong!!"
#endif

#include <Audio.h>
#include <FFT.h>
#include <SDHCI.h>
#include <float.h>
#include <DNNRT.h>
#include <Servo.h>

// -----------------------------------------------------------------------------
// Compile-time constants and global library objects
// -----------------------------------------------------------------------------
#define FFT_LEN 512
#define NNB_FILE "model.nnb"
#define BUFFERING_TIME_MS (FFT_LEN * 1000 / AS_SAMPLINGRATE_16000)
#define BUFFER_SIZE_BYTES (FFT_LEN * sizeof(int16_t) * AS_CHANNEL_STEREO)

FFTClass<AS_CHANNEL_STEREO, FFT_LEN> FFT;
AudioClass* theAudio = AudioClass::getInstance();
SDClass SD;
DNNRT dnnrt;

// =========================
// Types: Step, Sequence
// =========================
enum Level : int8_t {
  LEVEL_KEEP = -1,
  LEVEL_OFF  = 0,
  LEVEL_LOW,
  LEVEL_MID,
  LEVEL_HIGH,
  LEVEL_FULL
};

struct Step {
  int eyelidDeg;
  int pitchDeg;
  int yawDeg;
  uint16_t stepDelayMs;
  uint16_t holdMs;
  Level level;
};

struct Sequence {
  const Step* steps;
  int count;
};

// =========================
// SmoothServo
// =========================
struct SmoothServo {
  Servo s;
  int cur, target;
  uint16_t stepMs;
  unsigned long lastMs;
  SmoothServo(int initAngle = 90) : cur(initAngle), target(initAngle), stepMs(10), lastMs(0) {}
  void attach(int pin) { s.attach(pin); s.write(cur); }
  void setTarget(int t, uint16_t ms) { target = constrain(t, 0, 180); stepMs = (ms == 0) ? 1 : ms; }
  void update(unsigned long now) {
    if (cur == target) return;
    if ((now - lastMs) >= stepMs) {
      lastMs = now;
      int stepSize = 1;
      if (stepMs <= 1) stepSize = 12;
      else if (stepMs <= 3) stepSize = 8;
      else if (stepMs <= 8) stepSize = 4;
      else if (stepMs <= 20) stepSize = 2;
      int diff = target - cur;
      if (diff > 0) cur += min(stepSize, diff);
      else cur -= min(stepSize, -diff);
      cur = constrain(cur, 0, 180);
      s.write(cur);
    }
  }
  bool moving() const { return cur != target; }
};

// サーボ初期角度
SmoothServo Eyelid(112), Pitch(97), Yaw(95);

// =========================
// Level pins / table
// =========================
static const uint8_t kLevelPins[] = {4, 10, 11, 12, 13};
static const int kLevelPinCount = (int)(sizeof(kLevelPins) / sizeof(kLevelPins[0]));

struct LevelPattern { Level level; uint8_t pinState[5]; };
static const LevelPattern kLevelTable[] = {
  { LEVEL_FULL, {LOW, LOW,  HIGH, HIGH, HIGH} },
  { LEVEL_HIGH, {LOW, HIGH, LOW,  HIGH, HIGH} },
  { LEVEL_MID,  {LOW, HIGH, HIGH, LOW,  HIGH} },
  { LEVEL_LOW,  {LOW, HIGH, HIGH, HIGH, LOW } },
  { LEVEL_OFF,  {HIGH,  HIGH, HIGH, HIGH, HIGH} },
};
static Level g_currentLevel = LEVEL_OFF;

static inline void applyLevel(Level lv) {
  for (size_t t = 0; t < sizeof(kLevelTable) / sizeof(kLevelTable[0]); ++t) {
    if (kLevelTable[t].level != lv) continue;
    for (int i = 0; i < kLevelPinCount; ++i) digitalWrite(kLevelPins[i], kLevelTable[t].pinState[i]);
    g_currentLevel = lv;
    return;
  }
  // fallback
  for (int i = 0; i < kLevelPinCount; ++i) digitalWrite(kLevelPins[i], HIGH);
  digitalWrite(kLevelPins[0], LOW);
  g_currentLevel = LEVEL_OFF;
}

// =========================
// Sequences
// =========================
const Step seqA[] = {
  {100,97,95,50,50,LEVEL_KEEP},{85,87,95,70,0,LEVEL_LOW},{85,87,95,1,50,LEVEL_MID},
  {85,87,95,1,50,LEVEL_HIGH},{100,97,95,70,500,LEVEL_FULL},{85,87,95,70,0,LEVEL_LOW},
  {85,87,95,1,50,LEVEL_MID},{85,87,95,1,50,LEVEL_HIGH},{97,96,95,80,500,LEVEL_FULL},
  {88,91,95,120,700,LEVEL_HIGH},{88,91,95,1,50,LEVEL_MID},{88,91,95,1,50,LEVEL_LOW},
  {80,87,95,50,50,LEVEL_OFF}
};
const Step seqB[] = {
  {112,97,95,10,700,LEVEL_FULL},{90,97,95,1,70,LEVEL_FULL},{112,97,95,1,70,LEVEL_FULL},
  {90,97,95,1,70,LEVEL_FULL},{112,97,95,1,70,LEVEL_FULL}
};
const Step seqC[] = {
  {112,97,95,60,100,LEVEL_FULL},{112,90,95,60,100,LEVEL_FULL},{112,97,95,60,500,LEVEL_FULL},
  {104,97,95,60,500,LEVEL_HIGH},{104,97,95,50,100,LEVEL_MID},{104,97,95,1,1000,LEVEL_LOW}
};
const Step seqD[] = {
  {104,97,95,60,100,LEVEL_LOW},{104,90,95,60,100,LEVEL_LOW},{104,97,95,60,500,LEVEL_LOW},
  {112,97,95,40,500,LEVEL_MID},{112,97,95,50,100,LEVEL_HIGH},{112,97,95,1,100,LEVEL_FULL}
};
const Step seqE[] = {
  {102,97,95,15,100,LEVEL_OFF},{102,97,95,1,100,LEVEL_LOW},{102,97,95,1,100,LEVEL_MID},
  {102,97,95,1,100,LEVEL_HIGH},{102,97,95,1,100,LEVEL_FULL},{102,97,95,1,100,LEVEL_HIGH},
  {102,97,95,1,100,LEVEL_MID},{102,97,95,1,100,LEVEL_LOW},{102,97,95,1,100,LEVEL_MID},
  {102,97,95,1,100,LEVEL_HIGH},{102,97,95,1,100,LEVEL_FULL},{102,97,95,1,100,LEVEL_HIGH},
  {102,97,95,1,100,LEVEL_LOW},{102,97,95,1,100,LEVEL_MID},{102,97,95,1,100,LEVEL_HIGH},
  {102,97,95,1,100,LEVEL_FULL},{102,97,95,1,100,LEVEL_HIGH},{102,97,95,1,100,LEVEL_MID},
  {102,97,95,1,100,LEVEL_LOW},{102,97,95,1,100,LEVEL_MID},{102,97,95,1,100,LEVEL_HIGH},
  {102,97,95,1,100,LEVEL_FULL},{102,97,95,1,100,LEVEL_HIGH},{102,97,100,15,1,LEVEL_MID},
  {102,97,90,15,1,LEVEL_LOW},{102,97,100,15,1,LEVEL_LOW},{102,97,90,15,1,LEVEL_LOW},
  {102,97,95,15,70,LEVEL_LOW},{90,97,95,15,70,LEVEL_LOW},{102,97,95,15,70,LEVEL_LOW}
};
const Step seqF[] = {
  {80,87,95,100,500,LEVEL_KEEP},{80,87,100,100,50,LEVEL_KEEP},{80,87,90,50,50,LEVEL_KEEP},
  {80,87,100,50,50,LEVEL_KEEP},{80,87,90,50,50,LEVEL_KEEP},{80,87,95,90,400,LEVEL_KEEP},
  {80,87,95,90,50,LEVEL_OFF}
};
const Step seqG[] = {
  {112,97,95,50,100,LEVEL_LOW},{90,97,95,5,70,LEVEL_LOW},{112,97,95,5,70,LEVEL_FULL},
  {112,97,85,50,100,LEVEL_FULL},{112,90,85,60,400,LEVEL_FULL},{112,97,85,60,150,LEVEL_FULL},
  {112,97,95,55,1000,LEVEL_FULL}
};
const Step seqH[] = {
  {90,97,95,1,70,LEVEL_KEEP},{112,97,95,1,70,LEVEL_KEEP}
};
const Step seqI[] = {
  {90,97,95,2,70,LEVEL_KEEP},{104,97,95,2,70,LEVEL_KEEP}
};

const Sequence SEQA = { seqA, (int)(sizeof(seqA) / sizeof(seqA[0])) };
const Sequence SEQB = { seqB, (int)(sizeof(seqB) / sizeof(seqB[0])) };
const Sequence SEQC = { seqC, (int)(sizeof(seqC) / sizeof(seqC[0])) };
const Sequence SEQD = { seqD, (int)(sizeof(seqD) / sizeof(seqD[0])) };
const Sequence SEQE = { seqE, (int)(sizeof(seqE) / sizeof(seqE[0])) };
const Sequence SEQF = { seqF, (int)(sizeof(seqF) / sizeof(seqF[0])) };
const Sequence SEQG = { seqG, (int)(sizeof(seqG) / sizeof(seqG[0])) };
const Sequence SEQH = { seqH, (int)(sizeof(seqH) / sizeof(seqH[0])) };
const Sequence SEQI = { seqI, (int)(sizeof(seqI) / sizeof(seqI[0])) };

// =========================
// Runtime state
// =========================
const Sequence* activeSeq = nullptr;
int activeSeqId = -1;
int idx = 0;
unsigned long holdStartMs = 0;
bool holding = false;
bool running = false;
bool seqActive[9] = { false };

unsigned long lastInteractionMs = 0;
const unsigned long IDLE_TIMEOUT_MS = 60000UL;

const unsigned long DETECT_DEBOUNCE_MS = 800;
unsigned long lastDetectMs = 0;

int lastCompletedSeqId = -1;
unsigned long lastCompletedMs = 0;

// SEQH/SEQI を除く最後の完了時刻（SEQA 自動起動判定用）
unsigned long lastNonBlinkCompletedMs = 0;

// SEQH / SEQI periodic timers
unsigned long lastBlinkHMs = 0;
unsigned long lastBlinkIMs = 0;
const unsigned long BLINK_INTERVAL_MS = 15000UL; // 15s

// SEQA auto after last non-blink (180s)
const unsigned long AUTO_AFTER_LAST_MS = 180000UL;

static const char* kLabels[] = {"wake_up","good_night","brightly","darkly","clap","blow"};
static const int kNumLabels = 6;
static float labelThresholds[kNumLabels] = {0.50f,0.50f,0.50f,0.50f,0.50f,0.50f};

// spectrogram buffers
static const int frames = 40;
static const int fft_samples = 96;
static float spc_data[frames * fft_samples];
static float hist[frames];

static const uint32_t buffering_time = BUFFERING_TIME_MS;
static const uint32_t buffer_size = BUFFER_SIZE_BYTES;

static char buff[BUFFER_SIZE_BYTES];
static float pDstFG[FFT_LEN];
static float pDstBG[FFT_LEN];
static float pDst[FFT_LEN / 2];

// =========================
// Helpers: startStep, getSequenceById, startSequence, stopSequence
// =========================
void startStep(int i) {
  const Step& st = activeSeq->steps[i];
  Eyelid.setTarget(st.eyelidDeg, st.stepDelayMs);
  Pitch.setTarget(st.pitchDeg, st.stepDelayMs);
  Yaw.setTarget(st.yawDeg, st.stepDelayMs);
  if (st.level != LEVEL_KEEP) applyLevel(st.level);
  holding = false;
}

const Sequence* getSequenceById(int id) {
  switch (id) {
    case 0: return &SEQA;
    case 1: return &SEQB;
    case 2: return &SEQC;
    case 3: return &SEQD;
    case 4: return &SEQE;
    case 5: return &SEQF;
    case 6: return &SEQG;
    case 7: return &SEQH;
    case 8: return &SEQI;
    default: return nullptr;
  }
}

void startSequence(const Sequence* seq, int seqId) {
  if (running) return;

  // block rules (original mapping)
  bool blocked = false;
  switch (lastCompletedSeqId) {
    case 0: if (seqId==0||seqId==2||seqId==3||seqId==5||seqId==7||seqId==8) blocked = true; break;
    case 1: if (seqId==1||seqId==3||seqId==6||seqId==8) blocked = true; break;
    case 2: if (seqId==2||seqId==6||seqId==7) blocked = true; break;
    case 3: if (seqId==3||seqId==6||seqId==8) blocked = true; break;
    case 4: if (seqId==2||seqId==6||seqId==7) blocked = true; break;
    case 5: if (seqId==0||seqId==2||seqId==3||seqId==5||seqId==7||seqId==8) blocked = true; break;
    case 6: if (seqId==1||seqId==3||seqId==6||seqId==8) blocked = true; break;
    case 7: if (seqId==1||seqId==3||seqId==6||seqId==8) blocked = true; break;
    case 8: if (seqId==1||seqId==2||seqId==6||seqId==7) blocked = true; break;
    default: break;
  }
  if (blocked) return;

  activeSeq = seq;
  activeSeqId = seqId;
  idx = 0;
  running = true;
  if (seqId >= 0 && seqId < 9) seqActive[seqId] = true;
  startStep(idx);
  Serial.print("sequence started id=");
  Serial.println(seqId);
  lastInteractionMs = millis();
}

void stopSequence() {
  running = false;
  activeSeq = nullptr;
  holding = false;
  if (activeSeqId >= 0 && activeSeqId < 9) seqActive[activeSeqId] = false;
  activeSeqId = -1;
  for (int i = 0; i < 9; ++i) seqActive[i] = false;
  applyLevel(LEVEL_OFF);
  Serial.println("sequence stopped");
  lastInteractionMs = millis();
}

// =========================
// setup
// =========================
void setup() {
  for (int i = 0; i < kLevelPinCount; ++i) pinMode(kLevelPins[i], OUTPUT);
  // apply LEVEL_FULL at startup
  for (size_t t = 0; t < sizeof(kLevelTable) / sizeof(kLevelTable[0]); ++t) {
    if (kLevelTable[t].level != LEVEL_FULL) continue;
    for (int i = 0; i < kLevelPinCount; ++i) digitalWrite(kLevelPins[i], kLevelTable[t].pinState[i]);
    g_currentLevel = LEVEL_FULL;
    break;
  }

  Eyelid.attach(3);
  Pitch.attach(6);
  Yaw.attach(5);

  Serial.begin(115200);

  unsigned long sdStart = millis();
  const unsigned long SD_TIMEOUT_MS = 10000UL;
  while (!SD.begin()) {
    if (millis() - sdStart > SD_TIMEOUT_MS) {
      Serial.println("SD init timeout");
      delay(1000);
      sdStart = millis();
    }
  }

  File nnbfile = SD.open(NNB_FILE);
  if (!nnbfile) { Serial.println("nnb not found"); while (1) delay(1000); }
  int ret = dnnrt.begin(nnbfile);
  nnbfile.close();
  if (ret < 0) { Serial.println("DNN init error"); while (1) delay(1000); }

  FFT.begin(WindowHamming, AS_CHANNEL_STEREO, (FFT_LEN / 2));

  theAudio->begin();
  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);
  int err = theAudio->initRecorder(AS_CODECTYPE_PCM, "/mnt/sd0/BIN", AS_SAMPLINGRATE_16000, AS_CHANNEL_STEREO);
  if (err != AUDIOLIB_ECODE_OK) { Serial.println("Recorder init error"); while (1) delay(1000); }
  theAudio->startRecorder();

  memset(spc_data, 0, sizeof(spc_data));
  memset(hist, 0, sizeof(hist));

  lastInteractionMs = millis();
  lastBlinkHMs = millis();
  lastBlinkIMs = millis();
  lastNonBlinkCompletedMs = 0;
}

// =========================
// loop
// =========================
void loop() {
  unsigned long now = millis();

  // minimal serial commands
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') continue;
    lastInteractionMs = now;
    if (c == 's') { stopSequence(); continue; }
    if (c == 'r') {
      if (activeSeqId >= 0 && activeSeqId < 9) {
        const Sequence* seq = getSequenceById(activeSeqId);
        if (seq) startSequence(seq, activeSeqId);
      }
      continue;
    }
    if (!running) {
      if      (c == 'a') startSequence(&SEQA, 0);
      else if (c == 'b') startSequence(&SEQB, 1);
      else if (c == 'c') startSequence(&SEQC, 2);
      else if (c == 'd') startSequence(&SEQD, 3);
      else if (c == 'e') startSequence(&SEQE, 4);
      else if (c == 'f') startSequence(&SEQF, 5);
      else if (c == 'g') startSequence(&SEQG, 6);
      else if (c == 'h') startSequence(&SEQH, 7);
      else if (c == 'i') startSequence(&SEQI, 8);
    }
  }

  // SEQA auto-start: 45s after last non-blink completed
  if (!running && lastNonBlinkCompletedMs != 0) {
    if (now - lastNonBlinkCompletedMs >= AUTO_AFTER_LAST_MS) {
      startSequence(&SEQA, 0);
    }
  }

  // SEQH periodic auto-run (17s), skip if lastCompletedSeqId == 4 (blow)
  if (!running && (now - lastBlinkHMs >= BLINK_INTERVAL_MS) && lastCompletedSeqId != 4) {
    startSequence(&SEQH, 7);
    if (running && activeSeqId == 7) lastBlinkHMs = now;
  }

  // SEQI periodic auto-run (17s), skip if lastCompletedSeqId == 4 (blow)
  if (!running && (now - lastBlinkIMs >= BLINK_INTERVAL_MS) && lastCompletedSeqId != 4) {
    startSequence(&SEQI, 8);
    if (running && activeSeqId == 8) lastBlinkIMs = now;
  }

  // servo updates
  Eyelid.update(now);
  Pitch.update(now);
  Yaw.update(now);

  if (running) {
    if (Eyelid.moving() || Pitch.moving() || Yaw.moving()) {
      // wait for servos
    } else {
      if (!holding) { holding = true; holdStartMs = now; }
      if ((now - holdStartMs) >= (activeSeq ? activeSeq->steps[idx].holdMs : 0)) {
        if (activeSeq) {
          idx++;
          if (idx >= activeSeq->count) {
            if (activeSeqId >= 0 && activeSeqId < 9) seqActive[activeSeqId] = false;
            lastCompletedSeqId = activeSeqId;
            lastCompletedMs = now;
            // update lastNonBlinkCompletedMs only if not blink sequences (7 or 8)
            if (lastCompletedSeqId != 7 && lastCompletedSeqId != 8) {
              lastNonBlinkCompletedMs = now;
            }
            running = false;
            activeSeq = nullptr;
            activeSeqId = -1;
            Serial.print("sequence finished id=");
            Serial.println(lastCompletedSeqId);
            lastInteractionMs = now;
          } else {
            startStep(idx);
          }
        }
      }
    }
  }

  // ====== audio frame read ======
  uint32_t read_size = 0;
  int ret = theAudio->readFrames(buff, buffer_size, &read_size);
  if (ret != AUDIOLIB_ECODE_OK && ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
    while (1) delay(1000);
  }
  if (read_size < buffer_size) {
    delay(buffering_time);
    return;
  }

  // ====== FFT ======
  FFT.put((q15_t*)buff, FFT_LEN);
  FFT.get(pDstFG, 0);
  FFT.get(pDstBG, 1);

  // MIC-A/B diff + clamp
  const float alpha = 0.8f;
  for (int f = 0; f < FFT_LEN / 2; ++f) {
    float v = pDstFG[f] - alpha * pDstBG[f];
    pDst[f] = (v < 0.0f) ? 0.0f : v;
  }

  // smoothing (moving average)
  const int array_size = 4;
  static float pArray[array_size][FFT_LEN / 2];
  static int g_counter = 0;
  if (g_counter == array_size) g_counter = 0;
  for (int i = 0; i < FFT_LEN / 2; ++i) {
    pArray[g_counter][i] = pDst[i];
    float sum = 0;
    for (int j = 0; j < array_size; ++j) sum += pArray[j][i];
    pDst[i] = sum / array_size;
  }
  ++g_counter;

  // spectrogram shift
  memmove(spc_data, spc_data + fft_samples, (frames - 1) * fft_samples * sizeof(float));
  memmove(hist, hist + 1, (frames - 1) * sizeof(float));

  float sound_power_nc = 0.0f;
  for (int f = 0; f < FFT_LEN / 2; ++f) sound_power_nc += pDst[f];

  hist[frames - 1] = sound_power_nc;
  float* sp_last = spc_data + (frames - 1) * fft_samples;
  memcpy(sp_last, pDst, fft_samples * sizeof(float));

  // VAD (pre/target/post = 6/28/6)
  const float sound_th = 70.0f;
  const float silent_th = 10.0f;
  float pre_area = 0.0f, target_area = 0.0f, post_area = 0.0f;
  for (int t = 0; t < frames; ++t) {
    if (t < 6) pre_area += hist[t];
    else if (t < 6 + 28) target_area += hist[t];
    else post_area += hist[t];
  }

  if (pre_area < silent_th && target_area >= sound_th && post_area < silent_th) {
    unsigned long now3 = millis();
    if (now3 - lastDetectMs < DETECT_DEBOUNCE_MS) {
      // ignore
    } else {
      lastDetectMs = now3;
      memset(hist, 0, sizeof(hist));

      // prepare DNN input
      float spmax = -FLT_MAX, spmin = FLT_MAX;
      for (int n = 0; n < frames * fft_samples; ++n) {
        float v = spc_data[n];
        if (v > spmax) spmax = v;
        if (v < spmin) spmin = v;
      }
      float denom = spmax - spmin;
      if (denom < 1e-9f) denom = 1e-9f;

      DNNVariable input(28 * 48);
      float* din = input.data();
      const int t_begin = 6;
      const int t_end = 6 + 28;
      int bf = (fft_samples / 2) - 1;
      for (int f = 0; f < fft_samples; f += 2) {
        int bt = 0;
        for (int t = t_begin; t < t_end; ++t) {
          float v0 = (spc_data[fft_samples * t + f] - spmin) / denom;
          float v1 = (spc_data[fft_samples * t + f + 1] - spmin) / denom;
          float v01 = 0.5f * (v0 + v1);
          if (v01 < 0.f) v01 = 0.f;
          if (v01 > 1.f) v01 = 1.f;
          din[28 * bf + bt] = v01;
          ++bt;
        }
        --bf;
      }

      // inference
      dnnrt.inputVariable(input, 0);
      dnnrt.forward();
      DNNVariable output = dnnrt.outputVariable(0);

      int index = -1;
      float value = -1.0f;
      int out_size = output.size();
      float* out_data = output.data();
      if (out_size > 0) {
        index = 0;
        value = out_data[0];
        for (int i = 1; i < out_size; ++i) {
          if (out_data[i] > value) { value = out_data[i]; index = i; }
        }
        if (index < 0 || index >= kNumLabels) index = -1;
        else if (value < labelThresholds[index]) index = -1;
      }

      lastInteractionMs = now3;

      // minimal serial: detection result and sequence mapping
      if (index >= 0 && index < kNumLabels) {
        Serial.print(kLabels[index]);
        Serial.print(" : ");
        Serial.println(value, 6);
        switch (index) {
          case 0: startSequence(&SEQG, 6); break;
          case 1: startSequence(&SEQF, 5); break;
          case 2: startSequence(&SEQD, 3); break;
          case 3: startSequence(&SEQC, 2); break;
          case 4: startSequence(&SEQB, 1); break;
          case 5: startSequence(&SEQE, 4); break;
          default: break;
        }
      } else {
        Serial.print("class#");
        Serial.print(index);
        Serial.print(" : ");
        Serial.println(value, 6);
      }
    }
  }
}
