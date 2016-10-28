#include <arm_neon.h>

#define NEON_REDUCE_OP4(_Tpvec, _Tpnvec, scalartype, func, suffix) \
inline scalartype pairwise_##func(const scalartype* ptr) \
{ \
	_Tpvec##_t a = vld1q_##suffix(ptr); \
	_Tpnvec##_t b = vp##func##_##suffix(vget_low_##suffix(a), vget_high_##suffix(a)); \
	return (scalartype)vget_lane_##suffix(vp##func##_##suffix(b, b), 0); \
}

#define NEON_REDUCE_OP8(_Tpvec, _Tpnvec, scalartype, func, suffix) \
inline scalartype pairwise_##func(const scalartype* ptr) \
{ \
	_Tpvec##_t a = vld1q_##suffix(ptr); \
	_Tpnvec##_t b = vp##func##_##suffix(vget_low_##suffix(a), vget_high_##suffix(a)); \
	b = vp##func##_##suffix(b, b); \
	return (scalartype)vget_lane_##suffix(vp##func##_##suffix(b, b), 0); \
}

NEON_REDUCE_OP4(float32x4, float32x2, float, max, f32)
NEON_REDUCE_OP4(float32x4, float32x2, float, min, f32)
NEON_REDUCE_OP4(float32x4, float32x2, float, add, f32)
NEON_REDUCE_OP4(int32x4, int32x2, int, max, s32)
NEON_REDUCE_OP4(int32x4, int32x2, int, min, s32)
NEON_REDUCE_OP4(int32x4, int32x2, int, add, s32)
NEON_REDUCE_OP8(int16x8, int16x4, short, max, s16)
NEON_REDUCE_OP8(int16x8, int16x4, short, min, s16)
NEON_REDUCE_OP8(int16x8, int16x4, short, add, s16)
NEON_REDUCE_OP4(uint32x4, uint32x2, unsigned, max, u32)
NEON_REDUCE_OP4(uint32x4, uint32x2, unsigned, min, u32)
NEON_REDUCE_OP4(uint32x4, uint32x2, unsigned, add, u32)
NEON_REDUCE_OP8(uint16x8, uint16x4, ushort, max, u16)
NEON_REDUCE_OP8(uint16x8, uint16x4, ushort, min, u16)
NEON_REDUCE_OP8(uint16x8, uint16x4, ushort, add, u16)

