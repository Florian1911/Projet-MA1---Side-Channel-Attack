/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main_no_uart.c
  * @brief          : AES SCA – sans UART, sans USB.
  *                   Plaintexts pré-chargés en flash (plaintexts_data.h).
  *                   Trigger GPIO PB8 autour de chaque chiffrement.
  *                   Synchronisation : Python arme le scope, appui sur RESET.
  *
  * Câblage :
  *   - Générateur 3.3V → shunt (2.4Ω) → GND carte  (low-side, ChB)
  *   - PB8 → ChA (trigger)
  *   - USB DÉCONNECTÉ pendant l'acquisition
  ******************************************************************************
  */
/* USER CODE END Header */
#include "main.h"
#include <stdint.h>
#include <string.h>

/* Plaintexts générés par generate_plaintexts.py  →  Core/Inc/plaintexts_data.h */
#include "plaintexts_data.h"

/* Trigger GPIO ----------------------------------------------------------------*/
#define TRIG_GPIO_Port  GPIOB
#define TRIG_Pin        GPIO_PIN_8
#define TRIG_HIGH()     HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_SET)
#define TRIG_LOW()      HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_RESET)

/* Délai entre deux chiffrements (ms) -----------------------------------------
   Python doit re-armer le scope entre chaque capture en mode single-trigger,
   ou simplement laisser le scope en rapid-block.
   10 ms est une marge confortable.                                            */
#define INTER_TRACE_MS  10u

/* Nombre total de captures voulues.
   Peut dépasser N_PLAINTEXTS: les plaintexts sont alors rejoués en boucle. */
#define N_CAPTURES_TARGET 50000u

/* Clé AES-128 ----------------------------------------------------------------*/
static const uint8_t KEY_128[16] = {
  0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,
  0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
};

/* --- NAIVE AES-128 IMPLEMENTATION POUR SCA --- */
static const uint8_t naive_sbox[256] = {
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t naive_rcon[11] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };
#define AES_XTIME(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))

static void naive_aes128_key_expansion(const uint8_t* key, uint8_t* round_keys) {
  memcpy(round_keys, key, 16);
  int bytesGenerated = 16, rconIteration = 1;
  uint8_t temp[4];
  while (bytesGenerated < 176) {
    for(int i = 0; i < 4; i++) temp[i] = round_keys[bytesGenerated - 4 + i];
    if (bytesGenerated % 16 == 0) {
      uint8_t t = temp[0]; temp[0] = temp[1]; temp[1] = temp[2]; temp[2] = temp[3]; temp[3] = t;
      for(int i = 0; i < 4; i++) temp[i] = naive_sbox[temp[i]];
      temp[0] ^= naive_rcon[rconIteration++];
    }
    for(int i = 0; i < 4; i++) {
      round_keys[bytesGenerated] = round_keys[bytesGenerated - 16] ^ temp[i];
      bytesGenerated++;
    }
  }
}

static void naive_aes128_encrypt(const uint8_t* pt, const uint8_t* round_keys, uint8_t* ct) {
  /* volatile force le compilateur à séparer proprement les écritures en RAM, maximisant la fuite */
  volatile uint8_t state[16];
  uint8_t tmp[16];
  uint8_t a[4];

  for(int i=0; i<16; i++) state[i] = pt[i] ^ round_keys[i];

  for(int round=1; round<=9; round++) {
    for(int i=0; i<16; i++) { state[i] = naive_sbox[state[i]]; __NOP(); } // SubBytes
    tmp[0]=state[0]; tmp[4]=state[4]; tmp[8]=state[8];   tmp[12]=state[12]; // ShiftRows
    tmp[1]=state[5]; tmp[5]=state[9]; tmp[9]=state[13];  tmp[13]=state[1];
    tmp[2]=state[10];tmp[6]=state[14];tmp[10]=state[2];  tmp[14]=state[6];
    tmp[3]=state[15];tmp[7]=state[3]; tmp[11]=state[7];  tmp[15]=state[11];
    for(int i=0; i<4; i++) { // MixColumns
      for(int j=0; j<4; j++) a[j] = tmp[i*4+j];
      uint8_t t = a[0] ^ a[1] ^ a[2] ^ a[3];
      state[i*4+0] = a[0] ^ t ^ AES_XTIME(a[0] ^ a[1]);
      state[i*4+1] = a[1] ^ t ^ AES_XTIME(a[1] ^ a[2]);
      state[i*4+2] = a[2] ^ t ^ AES_XTIME(a[2] ^ a[3]);
      state[i*4+3] = a[3] ^ t ^ AES_XTIME(a[3] ^ a[0]);
    }
    for(int i=0; i<16; i++) state[i] ^= round_keys[round*16 + i]; // AddRoundKey
  }
  for(int i=0; i<16; i++) { state[i] = naive_sbox[state[i]]; __NOP(); }
  tmp[0]=state[0]; tmp[4]=state[4]; tmp[8]=state[8];   tmp[12]=state[12];
  tmp[1]=state[5]; tmp[5]=state[9]; tmp[9]=state[13];  tmp[13]=state[1];
  tmp[2]=state[10];tmp[6]=state[14];tmp[10]=state[2];  tmp[14]=state[6];
  tmp[3]=state[15];tmp[7]=state[3]; tmp[11]=state[7];  tmp[15]=state[11];
  for(int i=0; i<16; i++) ct[i] = tmp[i] ^ round_keys[160 + i];
}
/* --------------------------------------------- */

/* Prototypes -----------------------------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/* ---------------------------------------------------------------------------*/
int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();

  /* Dérivation de la clé (Key Expansion) faite une seule fois */
  uint8_t round_keys[176];
  naive_aes128_key_expansion(KEY_128, round_keys);

  /* ---- Signal "prêt" : 3 clignotements LED ---- */
  for (int b = 0; b < 3; b++) {
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    HAL_Delay(200);
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    HAL_Delay(200);
  }

  /* ---- Délai de démarrage : laisse Python armer le scope ----
     Python doit avoir appelé RunBlock AVANT ce délai.
     3 s >> temps de traitement Python.                          */
  HAL_Delay(3000);

  /* ---- Boucle principale : un trigger par plaintext ---- */
  uint8_t ct[16];
  volatile uint8_t state[16];  // state volatile pour que le compilateur n'optimise pas

  for (uint32_t i = 0; i < N_CAPTURES_TARGET; i++) {
    uint32_t pt_idx = i % N_PLAINTEXTS;
    memcpy((uint8_t*)state, PLAINTEXTS[pt_idx], 16); // plaintext rejoué en boucle si besoin

    HAL_Delay(INTER_TRACE_MS);   /* pause → scope se ré-arme en single-trigger */

    __disable_irq(); /* DÉSACTIVE LE SYSTICK POUR ÉLIMINER LE JITTER TEMPOREL */
    TRIG_HIGH();
    // petit délai pour que le spike PB8 disparaisse
    for (volatile int j=0; j<500; j++) __NOP();

    // ---- XOR key only: state[b] = pt[b] xor key[b] ----
    for(int b=0; b<16; b++) {
        state[b] = state[b] ^ round_keys[b];
        __NOP();  // anti-optimisation
    }
    TRIG_LOW();
    __enable_irq(); /* RÉACTIVE LES INTERRUPTIONS */

    // ---- finish encryption (optional, pas nécessaire pour CPA) ----
    memcpy(ct, (const uint8_t*)state, 16);
  }

  /* ---- Fin : LED fixe allumée ---- */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
  while (1) { /* done */ }
}

/* ---------------------------------------------------------------------------*/
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState       = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM       = 8;
  RCC_OscInitStruct.PLL.PLLN       = 336;
  RCC_OscInitStruct.PLL.PLLP       = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ       = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) { Error_Handler(); }
  }

  RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                   | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
    Error_Handler();
  }
}

/* ---------------------------------------------------------------------------*/
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /* LED LD2 */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  GPIO_InitStruct.Pin   = LD2_Pin;
  GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull  = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* Trigger PB8 */
  HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_RESET);
  GPIO_InitStruct.Pin   = TRIG_Pin;
  GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull  = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;  /* vitesse réduite → spike courant minimal */
  HAL_GPIO_Init(TRIG_GPIO_Port, &GPIO_InitStruct);
}

/* ---------------------------------------------------------------------------*/
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
    HAL_Delay(100);
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif
