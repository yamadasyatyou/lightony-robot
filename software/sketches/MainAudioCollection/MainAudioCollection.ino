/**
 * LIGHTONY ROBOT - Audio Data Collection Utility
 * * This sketch is used to collect voice data for training the neural network 
 * model used in the LIGHTONY robot.
 * * Acknowledgments:
 * This utility is based on the "MainAudioCollection.ino" sample from:
 * - Book: "SPRESENSEではじめるローパワーエッジAI" by Yoshinori Oota
 * - Repository: https://github.com/TE-YoshinoriOota/Spresense-LowPower-EdgeAI
 * * Specialized for capturing voice commands to improve LIGHTONY's 
 * recognition accuracy.
 */

#ifdef SUBCORE
#error "Core selection is wrong!!"
#endif

#include <Audio.h>
#include <FFT.h>

#define FFT_LEN 512

// ステレオ、512サンプルでFFTを初期化
FFTClass<AS_CHANNEL_STEREO, FFT_LEN> FFT;

AudioClass* theAudio = AudioClass::getInstance();

#include <MP.h>
#include <MPMutex.h>          // サブコア間の同期ライブラリ
MPMutex mutex(MP_MUTEX_ID0);
const int subcore = 1;        // サブコアの番号

#include <SDHCI.h>
SDClass SD;
File myFile;

#include <float.h> // FLT_MAX, FLT_MIN
#include <BmpImage.h>
BmpImage bmp;

// ===================== パラメータ定義 =====================
static const int frames         = 40;  // 時間方向（内部保持列数） 6+28+6
static const int fft_samples    = 96;  // 周波数方向（使用bin数 0..約3kHz想定）
static const int pre_frames     = 6;   // 無音プリ
static const int target_frames  = 28;  // 発話区間（抽出・保存対象）
static const int post_frames    = 6;   // 無音ポスト

static const int time_cols      = target_frames; // BMP横幅 28（=target）
static const int freq_rows      = 48;            // BMP縦幅 96→2bin平均で48

static const float alpha = 0.8f; // Rch抑圧係数（環境音抑圧）
static const float sound_th  = 70.f; // 発話しきい値（総エネルギー）
static const float silent_th = 10.f; // 無音しきい値

// ===================== 平滑化（4フレーム移動平均） =====================
void averageSmooth(float* dst) {
  const int array_size = 4;
  static float pArray[array_size][FFT_LEN/2];
  static int g_counter = 0;
  if (g_counter == array_size) g_counter = 0;
  for (int i = 0; i < FFT_LEN/2; ++i) {
    pArray[g_counter][i] = dst[i];
    float sum = 0;
    for (int j = 0; j < array_size; ++j) {
      sum += pArray[j][i];
    }
    dst[i] = sum / array_size;
  }
  ++g_counter;
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  while (!SD.begin()) { Serial.println("insert SD card"); }

  FFT.begin(WindowHamming, AS_CHANNEL_STEREO, (FFT_LEN/2));

  Serial.println("Init Audio Recorder");
  theAudio->begin();

  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);

  int err = theAudio->initRecorder(AS_CODECTYPE_PCM,
                                   "/mnt/sd0/BIN",
                                   AS_SAMPLINGRATE_16000,
                                   AS_CHANNEL_STEREO);
  if (err != AUDIOLIB_ECODE_OK) {
    Serial.println("Recorder initialize error");
    while (1);
  }

  Serial.println("Start Recorder");
  theAudio->startRecorder(); // 録音開始

  MP.begin(subcore); // サブコア開始
}

void loop() {
  static const uint32_t buffering_time =
      FFT_LEN * 1000 / AS_SAMPLINGRATE_16000; // ≒ 32ms
  static const uint32_t buffer_size =
      FFT_LEN * sizeof(int16_t) * AS_CHANNEL_STEREO; 

  static char  buff[buffer_size]; 
  static float pDstFG[FFT_LEN];   
  static float pDstBG[FFT_LEN];   
  static float pDst[FFT_LEN/2];   

  static float spc_data[frames * fft_samples];
  static float hist[frames];

  uint32_t read_size;
  int ret;

  ret = theAudio->readFrames(buff, buffer_size, &read_size);
  if (ret != AUDIOLIB_ECODE_OK &&
      ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
    Serial.println(String("Error err = ") + String(ret));
    theAudio->stopRecorder();
    while (1);
  }
  if (read_size < buffer_size) {
    delay(buffering_time);
    return;
  }


  FFT.put((q15_t*)buff, FFT_LEN);
  FFT.get(pDstFG, 0);  // MIC-A
  FFT.get(pDstBG, 1);  // MIC-B


  for (int f = 0; f < FFT_LEN/2; ++f) {
    float v = pDstFG[f] - alpha * pDstBG[f];
    pDst[f] = v < 0.f ? 0.f : v;
  }
  averageSmooth(pDst);


  for (int t = 1; t < frames; ++t) {
    float* dst = spc_data + (t-1) * fft_samples;
    float* src = spc_data +  t    * fft_samples;
    memcpy(dst, src, fft_samples * sizeof(float));
    hist[t-1] = hist[t];
  }


  float sound_power_nc = 0.f;
  for (int f = 0; f < FFT_LEN/2; ++f) sound_power_nc += pDst[f];
  hist[frames-1] = sound_power_nc;

  float* sp_last = spc_data + (frames-1) * fft_samples;
  memcpy(sp_last, pDst, fft_samples * sizeof(float));


  float pre_area = 0.f, target_area = 0.f, post_area = 0.f;
  for (int t = 0; t < frames; ++t) {
    if (t < pre_frames)                        pre_area    += hist[t];
    else if (t < pre_frames + target_frames)   target_area += hist[t];
    else                                       post_area   += hist[t];
  }

  if (pre_area < silent_th && target_area >= sound_th && post_area < silent_th) {

    memset(hist, 0, sizeof(hist));

    uint8_t bmp_data[time_cols * freq_rows];
    
    float spmax = FLT_MIN, spmin = FLT_MAX;
    for (int n = 0; n < frames * fft_samples; ++n) {
      float v = spc_data[n];
      if (v > spmax) spmax = v;
      if (v < spmin) spmin = v;
    }
    float denom = spmax - spmin;
    if (denom < 1e-9f) denom = 1e-9f; 

    int row = freq_rows - 1; 
    for (int f = 0; f < fft_samples; f += 2) {
      int col = 0; 
      for (int t = pre_frames; t < pre_frames + target_frames; ++t) {
        float v0 = (spc_data[fft_samples * t + f    ] - spmin) / denom;
        float v1 = (spc_data[fft_samples * t + f + 1] - spmin) / denom;
        float v  = (v0 + v1) * 0.5f * 255.f;
        if (v < 0.f)   v = 0.f;
        if (v > 255.f) v = 255.f;
        // バッファは行優先：index = 幅 * 行 + 列
        bmp_data[time_cols * row + col] = (uint8_t)(v + 0.5f);
        ++col;
      }
      --row;
    }

    digitalWrite(LED_BUILTIN, HIGH);   
    theAudio->stopRecorder();

    bool saved = false;
    do {
      static int n = 0;
      char fname[32]; memset(fname, 0, sizeof(fname));
      sprintf(fname, "%03d.bmp", n++);
      if (SD.exists(fname)) SD.remove(fname);

      File myFile = SD.open(fname, FILE_WRITE);
      if (!myFile) {
        Serial.println("SD open failed");
        break;
      }

      bmp.begin(BmpImage::BMP_IMAGE_GRAY8,
                time_cols,      
                freq_rows,      
                bmp_data);

      size_t expect_min = 2200; 
      size_t written = myFile.write(bmp.getBmpBuff(), bmp.getBmpSize());
      myFile.close();
      bmp.end();

      Serial.print("bmp size: "); Serial.println(bmp.getBmpSize());
      Serial.print("written:  "); Serial.println(written);
      Serial.println(String("save image as ") + fname);

      if (written > expect_min) saved = true; else Serial.println("Warning: written size too small");
    } while (0);

    theAudio->startRecorder();
    digitalWrite(LED_BUILTIN, LOW);    
    if (!saved) { Serial.println("save failed"); }
  }

  if (mutex.Trylock() != 0) return;
  int8_t sndid = 100;
  static const int disp_samples = 96;
  static float data[disp_samples];
  memcpy(data, pDst, disp_samples * sizeof(float));
  ret = MP.Send(sndid, &data, subcore);
  if (ret < 0) MPLog("FFT data Send Error\n");
  mutex.Unlock();
}
