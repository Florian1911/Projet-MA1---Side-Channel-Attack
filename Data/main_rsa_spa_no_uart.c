/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main_rsa_spa_no_uart.c
  * @brief          : RSA naive SPA target (no UART activity during capture)
  ******************************************************************************
  *
  * Notes:
  * - Intentionally variable-time square-and-multiply (SPA-friendly).
  * - Toy RSA parameters (small modulus) for side-channel demonstration only.
  * - Trigger PB8 wraps the full exponentiation.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdint.h>
#include "plaintexts_data.h"

#define TRIG_GPIO_Port GPIOB
#define TRIG_Pin       GPIO_PIN_8
#define TRIG_HIGH() HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_SET)
#define TRIG_LOW()  HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_RESET)

UART_HandleTypeDef huart2;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN 0 */
/* Toy RSA key (DO NOT USE IN REAL SYSTEMS) */
static const uint32_t RSA_N = 9173503u;   /* 3557 * 2579 */
static const uint32_t RSA_D = 6111579u;   /* inverse of e=3 mod phi */
static volatile uint32_t g_sink = 0u;
static volatile uint32_t g_zero_path_sink = 0u;

#define COMPILER_BARRIER() __asm volatile("" ::: "memory")

__attribute__((noinline))
static uint32_t mod_mul_u32(uint32_t a, uint32_t b, uint32_t mod)
{
  return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)mod);
}

/* Naive variable-time square-and-multiply (SPA-friendly by design). */
__attribute__((noinline))
static uint32_t rsa_modexp_naive_spa(uint32_t base, uint32_t exp, uint32_t mod)
{
  uint32_t r = 1u;
  base %= mod;

  /* Find MSB of exponent. */
  int msb = 31;
  while (msb > 0 && ((exp >> msb) & 1u) == 0u) {
    msb--;
  }

  for (int i = msb; i >= 0; i--) {
    /* Square always. */
    r = mod_mul_u32(r, r, mod);

    /* Slight deterministic workload to increase visibility. */
    for (volatile uint32_t d = 0; d < 24u; d++) {
      __asm volatile("nop");
    }

    /* Multiply only if bit=1 => SPA leakage source. */
    uint32_t bit = (exp >> i) & 1u;
    COMPILER_BARRIER();
    if (bit != 0u) {
      r = mod_mul_u32(r, base, mod);
      for (volatile uint32_t d = 0; d < 48u; d++) {
        __asm volatile("nop");
      }
    } else {
      /* Keep an explicit bit=0 path alive (anti-optimization). */
      g_zero_path_sink ^= (r ^ base ^ (uint32_t)i);
      for (volatile uint32_t d = 0; d < 12u; d++) {
        __asm volatile("nop");
      }
    }
    COMPILER_BARRIER();
  }

  return r;
}
/* USER CODE END 0 */

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init(); /* kept for project consistency, no UART traffic below */

  /* USER CODE BEGIN 2 */
  for (int b = 0; b < 3; b++) {
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    HAL_Delay(120);
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    HAL_Delay(120);
  }
  HAL_Delay(800);
  /* USER CODE END 2 */

  /* USER CODE BEGIN WHILE */
  for (uint32_t i = 0; i < N_PLAINTEXTS; i++) {
    /* Build a small message from plaintext bytes, then reduce mod N. */
    uint32_t m = ((uint32_t)PLAINTEXTS[i][0] << 8) | (uint32_t)PLAINTEXTS[i][1];
    m %= RSA_N;
    if (m == 0u) {
      m = 2u;
    }

    TRIG_HIGH();
    uint32_t c = rsa_modexp_naive_spa(m, RSA_D, RSA_N);
    TRIG_LOW();

    g_sink ^= c; /* avoid optimization-out */

    /* Outside trigger: give the scope time to re-arm. */
    for (volatile uint32_t d = 0; d < 120000u; d++) {
      __asm volatile("nop");
    }
  }

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
  while (1) { }
  /* USER CODE END WHILE */
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
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
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = TRIG_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(TRIG_GPIO_Port, &GPIO_InitStruct);
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
void assert_failed(uint8_t *file, uint32_t line)
{
  (void)file;
  (void)line;
}
#endif /* USE_FULL_ASSERT */
