/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main_attack_uart_sync.c
  * @brief          : AES SCA – sync via UART 'S' byte, pas de trigger GPIO.
  *                   High-side : ChA = 5V–shunt, ChB = shunt–carte.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
#include "main.h"
#include "mbedtls.h"
#include "mbedtls/aes.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN 0 */
static const uint8_t KEY_128[16] = {
  0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c
};

static const uint8_t PT_TEST[16] = {
  0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a
};

static const uint8_t CT_EXPECT[16] = {
  0x3a,0xd7,0x7b,0xb4,0x0d,0x7a,0x36,0x60,0xa8,0x9e,0xca,0xf3,0x24,0x66,0xef,0x97
};

static void uart_send_bytes(const uint8_t *buf, uint16_t len)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, len, HAL_MAX_DELAY);
}

static HAL_StatusTypeDef uart_recv_bytes(uint8_t *buf, uint16_t len)
{
  return HAL_UART_Receive(&huart2, buf, len, HAL_MAX_DELAY);
}

static void uart_recover(void)
{
  __HAL_UART_CLEAR_OREFLAG(&huart2);
  __HAL_UART_CLEAR_NEFLAG(&huart2);
  __HAL_UART_CLEAR_FEFLAG(&huart2);
  __HAL_UART_CLEAR_PEFLAG(&huart2);
}

static void uart_print_hex(UART_HandleTypeDef *huart, const char *label, const uint8_t *buf, size_t len)
{
  char byte_hex[3];
  HAL_UART_Transmit(huart, (uint8_t *)label, strlen(label), HAL_MAX_DELAY);
  for (size_t i = 0; i < len; i++) {
    snprintf(byte_hex, sizeof(byte_hex), "%02X", buf[i]);
    HAL_UART_Transmit(huart, (uint8_t *)byte_hex, 2, HAL_MAX_DELAY);
  }
  HAL_UART_Transmit(huart, (uint8_t *)"\r\n", 2, HAL_MAX_DELAY);
}
/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_MBEDTLS_Init();

  mbedtls_aes_context aes;
  uint8_t selftest_ct[16] = {0};
  uint8_t cmd = 0;
  uint8_t resp[17] = {0};
  uint8_t pt[16] = {0};
  uint8_t ct[16] = {0};

  mbedtls_aes_init(&aes);
  if (mbedtls_aes_setkey_enc(&aes, KEY_128, 128) != 0) {
    Error_Handler();
  }

  /* Self-test */
  if (mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT, PT_TEST, selftest_ct) != 0) {
    Error_Handler();
  }
  if (memcmp(selftest_ct, CT_EXPECT, sizeof(selftest_ct)) != 0) {
    uart_print_hex(&huart2, "AES FAIL got=", selftest_ct, sizeof(selftest_ct));
    uart_print_hex(&huart2, "AES exp=", CT_EXPECT, sizeof(CT_EXPECT));
    while (1) {
      HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
      HAL_Delay(100);
    }
  }
  uart_print_hex(&huart2, "AES OK ct=", selftest_ct, sizeof(selftest_ct));
  uart_send_bytes((const uint8_t *)"READY\r\n", 7);

  while (1)
  {
    /* Attendre commande 'P' */
    if (uart_recv_bytes(&cmd, 1) != HAL_OK) {
      uart_recover();
      continue;
    }
    if (cmd != 'P') {
      continue;
    }

    /* Recevoir le plaintext */
    if (uart_recv_bytes(pt, sizeof(pt)) != HAL_OK) {
      uart_recover();
      continue;
    }

    /* Signal de sync : envoyer 'S', puis attendre que le PC arme le Pico.
       Le PC reçoit 'S' (~87 µs UART + latence Python ~1-2 ms),
       arme le Pico (~1 ms), puis le délai ci-dessous garantit que
       le scope est prêt avant le début de l'AES.
       Régler SYNC_DELAY_MS selon la latence observée (5 ms = marge confortable). */
    uart_send_bytes((const uint8_t *)"S", 1);
    HAL_Delay(5);  /* 5 ms : laisse le temps au PC d'armer le Pico */

    /* Chiffrement AES (zone d'intérêt pour la mesure de courant) */
    if (mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT, pt, ct) != 0) {
      Error_Handler();
    }

    /* Renvoyer le ciphertext */
    resp[0] = 'C';
    memcpy(&resp[1], ct, sizeof(ct));
    if (HAL_UART_Transmit(&huart2, resp, sizeof(resp), HAL_MAX_DELAY) != HAL_OK) {
      uart_recover();
      continue;
    }
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
      Error_Handler();
    }
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* LED LD2 uniquement – pas de pin trigger GPIO */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

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
