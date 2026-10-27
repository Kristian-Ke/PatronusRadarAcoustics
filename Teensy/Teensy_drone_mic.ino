#include <Audio.h>

AudioInputI2SOct         i2s_oct;
AudioAnalyzeFFT1024      fft1;

AudioRecordQueue         queue1; AudioRecordQueue         queue2;
AudioRecordQueue         queue3; AudioRecordQueue         queue4;
AudioRecordQueue         queue5; AudioRecordQueue         queue6;
AudioRecordQueue         queue7; AudioRecordQueue         queue8;

AudioConnection          patchCord1(i2s_oct, 0, queue1, 0);
AudioConnection          patchCord2(i2s_oct, 1, queue2, 0);
AudioConnection          patchCord3(i2s_oct, 2, queue3, 0);
AudioConnection          patchCord4(i2s_oct, 3, queue4, 0);
AudioConnection          patchCord5(i2s_oct, 4, queue5, 0);
AudioConnection          patchCord6(i2s_oct, 5, queue6, 0);
AudioConnection          patchCord7(i2s_oct, 6, queue7, 0);
AudioConnection          patchCord8(i2s_oct, 7, queue8, 0);
AudioConnection          patchCord9(i2s_oct, 0, fft1, 0);

// --------------------------------
const bool STREAM_RAW_AUDIO   = true;   // true = record clips
const bool PRINT_SUMMARY      = false;
const bool DUMP_SPECTRUM      = false;   // true = print smoothed spectrum for plotting


const uint8_t SYNC_HEADER[4]  = {0xAA, 0x55, 0xAA, 0x55};

constexpr float FS            = AUDIO_SAMPLE_RATE_EXACT; 
constexpr float BIN_HZ        = FS / 1024.0f;             // ~43 Hz per FFT1024 bin
constexpr int   MAX_BIN       = 120;                      

constexpr float F0_MIN_HZ     = 130.0f;    // search range (blade-pass)
constexpr float F0_MAX_HZ     = 450.0f;
constexpr float F0_STEP_HZ    = 2.0f;     
constexpr int   MAX_HARM      = 8;

constexpr float SMOOTH        = 0.85f;     // spectrum averaging, higher = smoother but slower
constexpr float FLOOR_UP      = 0.002f;    // noise floor rises slowly
constexpr float FLOOR_DOWN    = 0.20f;     // and falls quickly 
constexpr float NOISE_EPS     = 4e-9f;     // avoids divide-by 0

constexpr float HARM_SNR      = 4.0f;      // harmonic must exceed floor by this ratio
constexpr int   MIN_HITS      = 4;         // harmonics that must be present
constexpr float MAX_ELEVATED_FRACTION = 0.5f; // veto broadband events

constexpr uint32_t DETECT_MS  = 500;       // must persist this long to be "stable"
constexpr uint32_t RELEASE_MS = 1000;      // and must be gone this long to release
constexpr int   PRINT_EVERY_N_UPDATES = 5;

// --------------------------------
static float S[MAX_BIN + 2];    // smoothed power per bin
static float Nf[MAX_BIN + 2];   // tracked noise floor per bin
static bool  spectrumInit = false;

struct Det {
  bool  updated = false;   // true only on loops where a new FFT frame arrived
  bool  frame   = false;
  bool  stable  = false;
  float f0 = 0.0f;
  float meanSnr = 0.0f;
  float elevated = 0.0f;
  int   hits = 0;
};
static Det det;

static inline float snrAt(int k) { return S[k] / (Nf[k] + NOISE_EPS); }

// returns the maximum SNR in a 3-bin window around the given center bin
static float windowPeakSnr(float centerBin) {
  int c = (int)(centerBin + 0.5f);
  float m = 0.0f;
  for (int k = c - 1; k <= c + 1; k++) {
    if (k < 1 || k > MAX_BIN) continue;
    float r = snrAt(k);
    if (r > m) m = r;
  }
  return m;
}

static void updateDetector() {
  static bool detecting = false;
  static uint32_t since = 0, lastSeen = 0;

  det.updated = false;
  if (!fft1.available()) return;   // state is kept avoiding reset between FFT frames
  det.updated = true;

  // smoothed spectrum
  for (int k = 1; k <= MAX_BIN; k++) {
    float m = fft1.read(k);
    float p = m * m;
    if (!spectrumInit) { S[k] = p; Nf[k] = p; }
    else               { S[k] = SMOOTH * S[k] + (1.0f - SMOOTH) * p; }
  }
  spectrumInit = true;

  // adaptive noise floor (frozen upward while a drone is being tracked)
  for (int k = 1; k <= MAX_BIN; k++) {
    if (S[k] < Nf[k])      Nf[k] += FLOOR_DOWN * (S[k] - Nf[k]);
    else if (!det.stable)  Nf[k] += FLOOR_UP   * (S[k] - Nf[k]);
  }

  // how much of the spectrum is elevated at all (broadband veto)
  int elevated = 0;
  for (int k = 2; k <= MAX_BIN; k++) if (snrAt(k) > HARM_SNR) elevated++;
  det.elevated = elevated / (float)(MAX_BIN - 1);

  // harmonic comb search over candidates
  int bestHits = 0; float bestMean = 0.0f, bestF0 = 0.0f;
  for (float f0 = F0_MIN_HZ; f0 <= F0_MAX_HZ; f0 += F0_STEP_HZ) {
    float spacing = f0 / BIN_HZ;
    int hits = 0; float sum = 0.0f;
    for (int h = 1; h <= MAX_HARM; h++) {
      float c = h * spacing;
      if (c + 1 > MAX_BIN) break;
      float r = windowPeakSnr(c);
      if (r >= HARM_SNR) { hits++; sum += r; }
    }
    float mean = hits ? sum / hits : 0.0f;
    if (hits > bestHits || (hits == bestHits && mean > bestMean)) {
      bestHits = hits; bestMean = mean; bestF0 = f0;
    }
  }
  det.hits = bestHits; det.meanSnr = bestMean; det.f0 = bestF0;
  det.frame = (bestHits >= MIN_HITS) && (det.elevated < MAX_ELEVATED_FRACTION);

  // time persistence 
  uint32_t now = millis();
  if (det.frame) {
    if (!detecting) { detecting = true; since = now; }
    lastSeen = now;
    det.stable = (now - since >= DETECT_MS);
  } else if (detecting && (now - lastSeen > RELEASE_MS)) {
    detecting = false;
    det.stable = false;
  }
}

void setup() {
  Serial.begin(115200);
  AudioMemory(150);

  queue1.begin(); queue2.begin(); queue3.begin(); queue4.begin();
  queue5.begin(); queue6.begin(); queue7.begin(); queue8.begin();

  Serial.println("Teensy drone detector debug sketch started");
}

void loop() {
  // Run the detector every pass, independent of the queues
  updateDetector();

  static uint32_t updates = 0;
  if (det.updated) {
    updates++;
    if (!STREAM_RAW_AUDIO && PRINT_SUMMARY && updates % PRINT_EVERY_N_UPDATES == 0) {
      Serial.print("f0=");        Serial.print(det.f0, 0);
      Serial.print(" hits=");     Serial.print(det.hits);
      Serial.print(" meanSnr=");  Serial.print(det.meanSnr, 1);
      Serial.print(" elevated="); Serial.print(det.elevated, 2);
      Serial.print(" frame=");    Serial.print(det.frame ? 1 : 0);
      Serial.print(" stable=");   Serial.println(det.stable ? 1 : 0);
    }
    if (!STREAM_RAW_AUDIO && DUMP_SPECTRUM && updates % PRINT_EVERY_N_UPDATES == 0) {
      Serial.print("SPEC,");
      for (int k = 1; k <= MAX_BIN; k++) { Serial.print(10.0f * log10f(S[k] + 1e-12f), 1);  Serial.print(','); }
      Serial.println();
      Serial.print("FLOOR,");
      for (int k = 1; k <= MAX_BIN; k++) { Serial.print(10.0f * log10f(Nf[k] + 1e-12f), 1); Serial.print(','); }
      Serial.println();
    }
  }

  if (queue1.available() >= 1 && queue2.available() >= 1 &&
      queue3.available() >= 1 && queue4.available() >= 1 &&
      queue5.available() >= 1 && queue6.available() >= 1 &&
      queue7.available() >= 1 && queue8.available() >= 1) {

    int16_t* b1 = queue1.readBuffer();
    int16_t* b2 = queue2.readBuffer();
    int16_t* b3 = queue3.readBuffer();
    int16_t* b4 = queue4.readBuffer();
    int16_t* b5 = queue5.readBuffer();
    int16_t* b6 = queue6.readBuffer();
    int16_t* b7 = queue7.readBuffer();
    int16_t* b8 = queue8.readBuffer();

    if (STREAM_RAW_AUDIO) {
      for (int i = 0; i < 128; i++) {
        b1[i] = b1[i] << 1;
        b2[i] = b2[i] << 1;
      }
      Serial.write(SYNC_HEADER, 4);
      Serial.write((uint8_t*)b1, 256);
      Serial.write((uint8_t*)b2, 256);
      Serial.write((uint8_t*)b3, 256);
      Serial.write((uint8_t*)b4, 256);
      Serial.write((uint8_t*)b5, 256);
      Serial.write((uint8_t*)b6, 256);
      Serial.write((uint8_t*)b7, 256);
      Serial.write((uint8_t*)b8, 256);
    }

    queue1.freeBuffer(); queue2.freeBuffer(); queue3.freeBuffer(); queue4.freeBuffer();
    queue5.freeBuffer(); queue6.freeBuffer(); queue7.freeBuffer(); queue8.freeBuffer();
  }
  else {
    if (queue1.available() > 6) {
      queue1.clear(); queue2.clear(); queue3.clear(); queue4.clear();
      queue5.clear(); queue6.clear(); queue7.clear(); queue8.clear();
    }
  }
}
