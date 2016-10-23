#include <iostream>
#include <arm_neon.h>
#ifndef __ARM_NEON
#error // activate neon
#endif

const uint64_t initState = 0x12345678;

typedef unsigned char uchar;
typedef unsigned short ushort;

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
};

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

#define NEON_REDUCE_OP(_Tpvec, _Tpnvec, scalartype, func, suffix) \
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
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned, max, u32)
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned, min, u32)
NEON_REDUCE_OP(uint32x4, uint32x2, unsigned, add, u32)
NEON_REDUCE_OP(uint16x8, uint16x4, ushort, max, u16)
NEON_REDUCE_OP(uint16x8, uint16x4, ushort, min, u16)
NEON_REDUCE_OP(uint16x8, uint16x4, ushort, add, u16)

#define NORMAL_REDUCE_OP_4(scalartype, func) \
inline scalartype normal_##func(const scalartype* ptr) \
{ \
	scalartype a0 = std::func(ptr[0], ptr[1]); \
	scalartype a1 = std::func(ptr[2], ptr[3]); \
	return std::func(a0, a1);  \
}
#define NORMAL_REDUCE_OP_8(scalartype, func) \
inline scalartype normal_##func(const scalartype* ptr) \
{ \
	scalartype a0 = std::func(ptr[0], ptr[1]); \
	scalartype a1 = std::func(ptr[2], ptr[3]); \
	scalartype a2 = std::func(ptr[4], ptr[5]); \
	scalartype a3 = std::func(ptr[6], ptr[7]); \
	a0 = std::func(a0, a1); \
	a1 = std::func(a2, a3); \
	return std::func(a0, a1);  \
}
#define NORMAL_REDUCE_ADD_4(scalartype) \
inline scalartype normal_add(const scalartype* ptr) \
{ \
	return (scalartype)(ptr[0] + ptr[1] + ptr[2] + ptr[3]); \
}
#define NORMAL_REDUCE_ADD_8(scalartype) \
inline scalartype normal_add(const scalartype* ptr) \
{ \
	return (scalartype)(ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7]); \
}

NORMAL_REDUCE_OP_4(int, max)
NORMAL_REDUCE_OP_4(int, min)
NORMAL_REDUCE_OP_4(unsigned, max)
NORMAL_REDUCE_OP_4(unsigned, min)
NORMAL_REDUCE_OP_8(short, max)
NORMAL_REDUCE_OP_8(short, min)
NORMAL_REDUCE_OP_8(ushort, max)
NORMAL_REDUCE_OP_8(ushort, min)

NORMAL_REDUCE_ADD_4(int)
NORMAL_REDUCE_ADD_4(unsigned)
NORMAL_REDUCE_ADD_8(short)
NORMAL_REDUCE_ADD_8(ushort)

template <typename T>
void testPairwise(RNG& rng, enum reduce_type reduce, int cIteration = 100)
{
	const int cElement = 16/sizeof(T);
	T buffer[cElement];

	for(int i = 0;i < cIteration;i++)
	{
		fillBuffer(rng, buffer);
		T resultNeon, resultNormal;
		switch(reduce)
		{
			case reduce_max:
				resultNeon = pairwise_max((const T*)buffer);
				resultNormal = normal_max((const T*)buffer);
				break;
			case reduce_min:
				resultNeon = pairwise_min((const T*)buffer);
				resultNormal = normal_min((const T*)buffer);
				break;
			case reduce_add:
				resultNeon = pairwise_add((const T*)buffer);
				resultNormal = normal_add((const T*)buffer);
				break;

		}
		if(resultNormal != resultNeon)
		{
			std::cout << "Mismatch type:" << reduce_str[reduce] << std::endl;
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
	testPairwise<int>(a, reduce_max, 10);
	testPairwise<int>(a, reduce_min, 10);
	testPairwise<short>(a, reduce_max, 10);
	testPairwise<short>(a, reduce_min, 10);
	return 0;
}
