#include <iostream>
#include <stdint.h>
#if defined _MSC_VER || defined __BORLANDC__
   typedef __int64 int64;
   typedef unsigned __int64 uint64;
#  define CV_BIG_INT(n)   n##I64
#  define CV_BIG_UINT(n)  n##UI64
#else
   typedef int64_t int64;
   typedef uint64_t uint64;
#  define CV_BIG_INT(n)   n##LL
#  define CV_BIG_UINT(n)  n##ULL
#endif

const uint64 initState = 0x12345678;
typedef unsigned char uchar;
typedef unsigned short ushort;

#if defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#include "simd_neon.h"
#elif defined __SSE2__ || defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#include "simd_sse.h"
#endif

class RNG
{
public:
	uint64 state;
	RNG()                        { state = 0xffffffff; };
	RNG(uint64 _state)           { state = _state ? _state : 0xffffffff; };
	operator unsigned char()     { return (unsigned char)next(); };
	operator char()              { return (char)next(); };
	operator unsigned short()    { return (unsigned short)next(); };
	operator short()             { return (short)next(); };
	operator int()               { return (int)next(); };
	operator unsigned()          { return next(); };

	inline unsigned next()
	{
	    state = (uint64)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
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

template <typename T, int mask>
void testPairwise(RNG& rng, enum reduce_type reduce, int cIteration)
{
	const int cElement = 16/sizeof(T);
	T buffer[cElement];

	for(int i = 0;i < cIteration;i++)
	{
		fillBuffer(rng, buffer);
		T resultSimd, resultNormal;
		switch(reduce)
		{
			case reduce_max:
				resultSimd = pairwise_max((const T*)buffer);
				resultNormal = normal_max((const T*)buffer);
				break;
			case reduce_min:
				resultSimd = pairwise_min((const T*)buffer);
				resultNormal = normal_min((const T*)buffer);
				break;
			case reduce_add:
                for(int j = 0;j < cElement;j++)
                {
                    buffer[j] &= mask;
                }
				resultSimd = pairwise_add((const T*)buffer);
				resultNormal = normal_add((const T*)buffer);
				break;

		}
		if(resultNormal != resultSimd)
		{
			std::cout << "Mismatch type:" << reduce_str[reduce] << std::endl;
			dumpArray(buffer);
			std::cout << std::endl;
			std::cout << "result SIMD  :" << resultSimd   << std::endl;
			std::cout << "result Normal:" << resultNormal << std::endl;
		}
	}
}

template <typename T>
void testPairwise(RNG& rng, enum reduce_type reduce, int cIteration = 100);

template <> void testPairwise<int>(RNG& rng, enum reduce_type reduce, int cIteration)
{
    testPairwise<int, 0x1fffffff>(rng, reduce, cIteration);
}
template <> void testPairwise<unsigned>(RNG& rng, enum reduce_type reduce, int cIteration)
{
    testPairwise<unsigned, 0x1fffffff>(rng, reduce, cIteration);
}
template <> void testPairwise<short>(RNG& rng, enum reduce_type reduce, int cIteration)
{
    testPairwise<short, 0xfff>(rng, reduce, cIteration);
}
template <> void testPairwise<unsigned short>(RNG& rng, enum reduce_type reduce, int cIteration)
{
    testPairwise<unsigned short, 0xfff>(rng, reduce, cIteration);
}

int main(int argc, char** argv)
{
	RNG a(initState);
	testPairwise<int>(a, reduce_max, 10);
	testPairwise<int>(a, reduce_min, 10);
	testPairwise<int>(a, reduce_add, 10);
	testPairwise<unsigned int>(a, reduce_max, 10);
	testPairwise<unsigned int>(a, reduce_min, 10);
	testPairwise<unsigned int>(a, reduce_add, 10);
	testPairwise<short>(a, reduce_max, 10);
	testPairwise<short>(a, reduce_min, 10);
	testPairwise<short>(a, reduce_add, 10);
	testPairwise<unsigned short>(a, reduce_max, 10);
	testPairwise<unsigned short>(a, reduce_min, 10);
	testPairwise<unsigned short>(a, reduce_add, 10);
	return 0;
}
