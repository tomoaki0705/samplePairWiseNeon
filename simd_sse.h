#include <emmintrin.h>

#define SSE_REDUCE_OP8(scalartype, func, sbit) \
inline scalartype pairwise_##func(const scalartype* ptr) \
{ \
	__m128i val; \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_##func##_epi16(val, _mm_srli_si128(val,8)); \
	val = _mm_##func##_epi16(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_epi16(val, _mm_srli_si128(val,2)); \
	return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype pairwise_##func(const unsigned scalartype* ptr) \
{ \
	__m128i val, smask; \
	smask = _mm_set1_epi16(sbit); \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_xor_si128(val, smask); \
	val = _mm_##func##_epi16(val, _mm_srli_si128(val,8)); \
	val = _mm_##func##_epi16(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_epi16(val, _mm_srli_si128(val,2)); \
	return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^ sbit); \
}

#define SSE_REDUCE_OP8_SUM(scalartype, suffix) \
inline scalartype pairwise_add(const scalartype* ptr) \
{ \
	__m128i val; \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_adds_##suffix(val, _mm_srli_si128(val,8)); \
	val = _mm_adds_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_adds_##suffix(val, _mm_srli_si128(val,2)); \
	return (scalartype)(_mm_cvtsi128_si32(val) & 0xffff); \
}

#define SSE_REDUCE_OP4(scalartype, func, sbit) \
inline scalartype pairwise_##func(const scalartype* ptr) \
{ \
	__m128i val; \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_##func##_epi32(val, _mm_srli_si128(val,8)); \
	val = _mm_##func##_epi32(val, _mm_srli_si128(val,4)); \
	return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype pairwise_##func(const unsigned scalartype* ptr) \
{ \
	__m128i val, smask; \
	smask = _mm_set1_epi32(sbit); \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_xor_si128(val, smask); \
	val = _mm_##func##_epi32(val, _mm_srli_si128(val,8)); \
	val = _mm_##func##_epi32(val, _mm_srli_si128(val,4)); \
	return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^ sbit); \
}

#define SSE_REDUCE_OP4_SUM(scalartype, func, suffix) \
inline scalartype pairwise_add(const scalartype* ptr) \
{ \
	__m128i val; \
	val = _mm_loadu_si128((const __m128i*)ptr); \
	val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
	return (scalartype)_mm_cvtsi128_si32(val); \
}

SSE_REDUCE_OP8_SUM(short, epi16)
SSE_REDUCE_OP8_SUM(unsigned short, epu16)
SSE_REDUCE_OP8(short, max, (short)-32768)
SSE_REDUCE_OP8(short, min, (short)-32768)

SSE_REDUCE_OP4_SUM(int, add, epi32)
SSE_REDUCE_OP4_SUM(unsigned, add, epi32)
SSE_REDUCE_OP4(int, max, 0x80000000)
SSE_REDUCE_OP4(int, min, 0x80000000)

