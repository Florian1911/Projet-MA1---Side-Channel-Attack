/*
 * Template firmware STM32 pour acquisition SCA AES (pas d'UART runtime).
 * - Trigger sur PB8 (monte avant AES, redescend apres AES)
 * - Utilise plaintexts_data.h genere cote PC
 */

#include "main.h"
#include "mbedtls.h"
#include "mbedtls/aes.h"
#include "plaintexts_data.h"

#include <stdint.h>
#include <string.h>

#define TRIG_GPIO_Port GPIOB
#define TRIG_Pin GPIO_PIN_8
#define TRIG_HIGH() HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_SET)
#define TRIG_LOW() HAL_GPIO_WritePin(TRIG_GPIO_Port, TRIG_Pin, GPIO_PIN_RESET)

static const uint8_t KEY_128[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_MBEDTLS_Init();

    mbedtls_aes_context aes;
    uint8_t ct[16] = {0};

    mbedtls_aes_init(&aes);
    if (mbedtls_aes_setkey_enc(&aes, KEY_128, 128) != 0) {
        Error_Handler();
    }

    for (int i = 0; i < 3; i++) {
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
        HAL_Delay(200);
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
        HAL_Delay(200);
    }
    HAL_Delay(3000);

    for (uint32_t i = 0; i < N_PLAINTEXTS; i++) {
        const uint8_t *pt = PLAINTEXTS[i];

        TRIG_HIGH();
        if (mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT, pt, ct) != 0) {
            TRIG_LOW();
            Error_Handler();
        }
        TRIG_LOW();

        for (volatile uint32_t d = 0; d < 150000; d++) {
            __asm volatile("nop");
        }
    }

    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    while (1) {
    }
}
