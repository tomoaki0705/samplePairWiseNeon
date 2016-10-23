#include <iostream>
#include <math.h>
#include <arm_neon.h>
#ifndef __ARM_NEON
#error // activate neon
#endif

const uint64_t initState = 0x12345678;

class RNG
{
public:
	uint64_t state;
	RNG()                        { state = 0xffffffff; };
	RNG(uint64_t _state)         { state = _state ? _state : 0xffffffff; };
	operator unsigned char()     { return (unsigned char)next(); };
	operator char()              { return (char)next(); };
	operator unsigned short()    { return (unsigned short)next(); };
	operator short()             { return (short)next(); };
	operator int()               { return (int)next(); };
	operator unsigned()          { return next(); };

	inline unsigned next()
	{
	    state = (uint64_t)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
	    return (unsigned)state;
	}
};	

enum reduce_type {
	reduce_max,
	reduce_min,
	reduce_add,
};

const char* reduce_str[] =
{
	"reduce_max",
	"reduce_min",
	"reduce_add",
}

template <typename T>
void fillBuffer(RNG& rng, T* buffer)
{
	const int cElement = 16/sizeof(T);
	for(int i = 0;i < cElement;i++)
	{
		buffer[i] = (T)rng;
	}
}

template <typename T>
void dumpArray(const T *array)
{
	const int cElement = 16/sizeof(T);
	for(int i = 0;i < cElement;i++)
	{
		std::cout << '\t' << array[i];
	}
}



short pairwise_max(const short* ptr)
{
	int16x8_t a = vld1q_s16(ptr);
	int16x4_t b = vpmax_s16(vget_low_s16(a), vget_high_s16(a));
	b = vpmax_s16(b, b);
	return (short)vget_lane_s16(vpmax_s16(b, b), 0);
}

NEON_REDUCE_OP(_Tpvec, _Tpnvec, scalartype, func, suffix) \
inline scalartype pairwise_##func(const scalartype* ptr) \
{ \
	_Tpvec##_t a = vld1q_##suffix(ptr); \
	_Tpnvec##_t b = vp##func##_##suffix(vget_low_##suffix(a), vget_high_##suffix(a)); \
	b = vp##func##_##suffix(b, b); \
	return (scalartype)vget_lane_##suffix(vp##func##_##suffix(b, b), 0); \
}

NEON_REDUCE_OP(float32x4, float32x2, float, max, f32)
NEON_REDUCE_OP(float32x4, float32x2, float, min, f32)
NEON_REDUCE_OP(float32x4, float32x2, float, add, f32)
NEON_REDUCE_OP(int32x4, int32x2, int, max, s32)
NEON_REDUCE_OP(int32x4, int32x2, int, min, s32)
NEON_REDUCE_OP(int32x4, int32x2, int, add, s32)
NEON_REDUCE_OP(int16x8, int16x4, short, max, s16)
NEON_REDUCE_OP(int16x8, int16x4, short, min, s16)
NEON_REDUCE_OP(int16x8, int16x4, short, add, s16)
NEON_REDUCE_OP(int8x16, int8x8, char, max, s8)
NEON_REDUCE_OP(int8x16, int8x8, char, min, s8)
NEON_REDUCE_OP(int8x16, int8x8, char, add, s8)
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned int, max, u32)
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned int, min, u32)
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned int, add, u32)
NEON_REDUCE_OP(uint16x8, uint16x4, unsigned short, max, u16)
NEON_REDUCE_OP(uint16x8, uint16x4, unsigned short, min, u16)
NEON_REDUCE_OP(uint16x8, uint16x4, unsigned short, add, u16)
NEON_REDUCE_OP(uint8x16, uint8x8, unsigned char, max, u8)
NEON_REDUCE_OP(uint8x16, uint8x8, unsigned char, min, u8)
NEON_REDUCE_OP(uint8x16, uint8x8, unsigned char, add, u8)

template <typename T>
void testPairwise(RNG& rng, enum reduce_type, int cIteration = 100)
{
	const int cElement = 16/sizeof(T);
	T buffer[cElement];

	for(int i = 0;i < cIteration;i++)
	{
		fillBuffer(rng, buffer);
		T resultNeon, resultNormal;
		switch(reduce_type)
		{
			case reduce_max:
				resultNeon = pairwise_max(buffer);
				resultNormal = normal_max(buffer);
				break;
			case reduce_min:
				resultNeon = pairwise_min(buffer);
				resultNormal = normal_min(buffer);
				break;
			case reduce_add:
				resultNeon = pairwise_add(buffer);
				resultNormal = normal_add(buffer);
				break;

		}
		if(resultNormal != resultNeon)
		{
			std::cout << "Mismatch type:" << reduce_str[reduce_type] << std::endl;
			dumpArray(buffer);
			std::cout << std::endl;
			std::cout << "result Neon  :" << resultNeon   << std::endl;
			std::cout << "result Normal:" << resultNormal << std::endl;
		}
	}
}


int main(int argc, char** argv)
{
	RNG a(initState);
	testPairwise<int>(a, reduce_max);
	testPairwise<int>(a, reduce_min);
	testPairwise<shrot>(a, reduce_add);
	testPairwise<shrot>(a, reduce_max);
	testPairwise<shrot>(a, reduce_min);
	testPairwise<char>(a, reduce_add);
	testPairwise<char>(a, reduce_max);
	testPairwise<char>(a, reduce_min);
	testPairwise<float>(a, reduce_add);
	testPairwise<float>(a, reduce_max);
	testPairwise<float>(a, reduce_min);
	return 0;
}
