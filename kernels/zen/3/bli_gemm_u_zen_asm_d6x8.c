/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#include "immintrin.h"
#include "xmmintrin.h"

void bli_dgemmt_u_zen_asm_6x8
     (
       dim_t               k0,
       dim_t               m_off,
       dim_t               n_off,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    //void*   a_next = bli_auxinfo_next_a( data );
    //void*   b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    double* restrict a0 = a;
    double* restrict b0 = b;
    double* restrict c0 = c;

    //double beta_val = *beta;

    double f_temp[8] __attribute__((aligned(64))) = {0.0};

    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    dim_t i,j,k;

    ymm0 = _mm256_loadu_pd((double const *)b0 + 0);
    ymm1 = _mm256_loadu_pd((double const *)b0 + 4);

    ymm4 = _mm256_setzero_pd();
    ymm5 = _mm256_setzero_pd();
    ymm6 = _mm256_setzero_pd();
    ymm7 = _mm256_setzero_pd();

    for( k = 0; k < k_iter; k++ )
    {
        //iteration 0
        ymm2 = _mm256_broadcast_sd((double const *)a0 + 0);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 1);

        ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 2);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 3);

        ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 4);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 5);

        ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

        ymm0 = _mm256_loadu_pd((double const *)b0 + 1*8 + 0);
        ymm1 = _mm256_loadu_pd((double const *)b0 + 1*8 + 4);

        //iteration 1
        ymm2 = _mm256_broadcast_sd((double const *)a0 + 6);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 7);

        ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 8);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 9);

        ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 10);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 11);

        ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

        ymm0 = _mm256_loadu_pd((double const *)b0 + 2*8 + 0);
        ymm1 = _mm256_loadu_pd((double const *)b0 + 2*8 + 4);

        //iteration 2
        ymm2 = _mm256_broadcast_sd((double const *)a0 + 12);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 13);

        ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 14);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 15);

        ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 16);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 17);

        ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

        ymm0 = _mm256_loadu_pd((double const *)b0 + 3*8 + 0);
        ymm1 = _mm256_loadu_pd((double const *)b0 + 3*8 + 4);

        //iteration 3
        ymm2 = _mm256_broadcast_sd((double const *)a0 + 18);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 19);

        ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 20);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 21);

        ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 22);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 23);

        ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

        a0 += 6*4;
        b0 += 8*4;

        ymm0 = _mm256_loadu_pd((double const *)b0 + 0*8 + 0);
        ymm1 = _mm256_loadu_pd((double const *)b0 + 0*8 + 4);

    }
    
    for(k = 0; k < k_left; k++)
    {
        ymm2 = _mm256_broadcast_sd((double const *)a0 + 0);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 1);

        ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 2);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 3);

        ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_sd((double const *)a0 + 4);
        ymm3 = _mm256_broadcast_sd((double const *)a0 + 5);

        ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

        a0 += 6;
        b0 += 8;

        ymm0 = _mm256_loadu_pd((double const*)(b0 + 0));
        ymm1 = _mm256_loadu_pd((double const*)(b0 + 4));
    }


    ymm0 = _mm256_broadcast_sd((double const *)alpha);

    ymm4 = _mm256_mul_pd(ymm0, ymm4);
    ymm5 = _mm256_mul_pd(ymm0, ymm5);
    ymm6 = _mm256_mul_pd(ymm0, ymm6);
    ymm7 = _mm256_mul_pd(ymm0, ymm7);
    ymm8 = _mm256_mul_pd(ymm0, ymm8);
    ymm9 = _mm256_mul_pd(ymm0, ymm9);
    ymm10 = _mm256_mul_pd(ymm0, ymm10);
    ymm11 = _mm256_mul_pd(ymm0, ymm11);
    ymm12 = _mm256_mul_pd(ymm0, ymm12);
    ymm13 = _mm256_mul_pd(ymm0, ymm13);
    ymm14 = _mm256_mul_pd(ymm0, ymm14);
    ymm15 = _mm256_mul_pd(ymm0, ymm15);

    double beta_val = *beta;
    if( 1 == cs_c ) //row-stored
    {
        if(beta_val != 0.0)
        {

           //1st row
            _mm256_storeu_pd(f_temp, ymm4);
            _mm256_storeu_pd(f_temp+4, ymm5);

            for( j = 0; j < 8; j++ )
                if(( m_off + 0 ) <= ( n_off + j))
                    c0[ 0 * rs_c + j ] = c[ 0 * rs_c + j ] * beta_val + f_temp[j];

            //2nd row
            _mm256_storeu_pd(f_temp, ymm6);
            _mm256_storeu_pd(f_temp+4, ymm7);

            for( j = 0; j < 8; j++ )
                if(( m_off + 1 ) <= ( n_off + j))
                    c0[ 1 * rs_c + j ] = c[ 1 * rs_c + j ] * beta_val + f_temp[j];


            //3rd row
            _mm256_storeu_pd(f_temp, ymm8);
            _mm256_storeu_pd(f_temp+4, ymm9);

            for( j = 0; j < 8; j++ )
                if(( m_off + 2 ) <= ( n_off + j))
                    c0[ 2 * rs_c + j ] = c[ 2 * rs_c + j ] * beta_val + f_temp[j];

            //4th row
            _mm256_storeu_pd(f_temp, ymm10);
            _mm256_storeu_pd(f_temp+4, ymm11);

            for( j = 0; j < 8; j++ )
                if(( m_off + 3 ) <= ( n_off + j))
                    c0[ 3 * rs_c + j ] = c[ 3 * rs_c + j ] * beta_val + f_temp[j];

            //5th row
            _mm256_storeu_pd(f_temp, ymm12);
            _mm256_storeu_pd(f_temp+4, ymm13);

            for( j = 0; j < 8; j++ )
                if(( m_off + 4 ) <= ( n_off + j))
                    c0[ 4 * rs_c + j ] = c[ 4 * rs_c + j ] * beta_val + f_temp[j];

            //6th row
            _mm256_storeu_pd(f_temp, ymm14);
            _mm256_storeu_pd(f_temp+4, ymm15);

            for( j = 0; j < 8; j++ )
                if(( m_off + 5 ) <= ( n_off + j))
                    c0[ 5 * rs_c + j ] = c[ 5 * rs_c + j ] * beta_val + f_temp[j];


        }
        else
        {
            //1st row
            _mm256_storeu_pd(f_temp, ymm4);
            _mm256_storeu_pd(f_temp+4, ymm5);

            for( j = 0; j < 8; j++ )
                if(( m_off + 0 ) <= ( n_off + j))
                    c0[ 0 * rs_c + j ] = f_temp[j];

            //2nd row
            _mm256_storeu_pd(f_temp, ymm6);
            _mm256_storeu_pd(f_temp+4, ymm7);

            for( j = 0; j < 8; j++ )
                if(( m_off + 1 ) <= ( n_off + j))
                    c0[ 1 * rs_c + j ] = f_temp[j];


            //3rd row
            _mm256_storeu_pd(f_temp, ymm8);
            _mm256_storeu_pd(f_temp+4, ymm9);

            for( j = 0; j < 8; j++ )
                if(( m_off + 2 ) <= ( n_off + j))
                    c0[ 2 * rs_c + j ] = f_temp[j];

            //4th row
            _mm256_storeu_pd(f_temp, ymm10);
            _mm256_storeu_pd(f_temp+4, ymm11);

            for( j = 0; j < 8; j++ )
                if(( m_off + 3 ) <= ( n_off + j))
                    c0[ 3 * rs_c + j ] = f_temp[j];

            //5th row
            _mm256_storeu_pd(f_temp, ymm12);
            _mm256_storeu_pd(f_temp+4, ymm13);

            for( j = 0; j < 8; j++ )
                if(( m_off + 4 ) <= ( n_off + j))
                    c0[ 4 * rs_c + j ] = f_temp[j];

            //6th row
            _mm256_storeu_pd(f_temp, ymm14);
            _mm256_storeu_pd(f_temp+4, ymm15);

            for( j = 0; j < 8; j++ )
                if(( m_off + 5 ) <= ( n_off + j))
                    c0[ 5 * rs_c + j ] = f_temp[j];
        }
        return;
    }
    else if( 1 == rs_c )//column stored
    {
        ymm2 = _mm256_unpacklo_pd(ymm4, ymm6);
        ymm3 = _mm256_unpacklo_pd(ymm8, ymm10);

        ymm0 = _mm256_permute2f128_pd(ymm2, ymm3, 0x20);
        ymm1 = _mm256_permute2f128_pd(ymm2, ymm3, 0x31);

        ymm4 = _mm256_unpackhi_pd(ymm4, ymm6);
        ymm8 = _mm256_unpackhi_pd(ymm8, ymm10);

        ymm2 = _mm256_permute2f128_pd(ymm4, ymm8, 0x20);
        ymm3 = _mm256_permute2f128_pd(ymm4, ymm8, 0x31);

        ymm8 = _mm256_unpacklo_pd(ymm5, ymm7);
        ymm10 = _mm256_unpacklo_pd(ymm9, ymm11);

        ymm4 = _mm256_permute2f128_pd(ymm8, ymm10, 0x20);
        ymm6 = _mm256_permute2f128_pd(ymm8, ymm10, 0x31);

        ymm5 = _mm256_unpackhi_pd(ymm5, ymm7);
        ymm9 = _mm256_unpackhi_pd(ymm9, ymm11);

        ymm8 = _mm256_permute2f128_pd(ymm5, ymm9, 0x20);
        ymm10 = _mm256_permute2f128_pd(ymm5, ymm9, 0x31);
      
        ymm9 = _mm256_unpacklo_pd(ymm12, ymm14);
        ymm11 = _mm256_unpacklo_pd(ymm13, ymm15);

        ymm5 = _mm256_permute2f128_pd(ymm9, ymm11, 0x20);
        ymm7 = _mm256_permute2f128_pd(ymm9, ymm11, 0x31);

        ymm12 = _mm256_unpackhi_pd(ymm12, ymm14);
        ymm14 = _mm256_unpackhi_pd(ymm13, ymm15);

        ymm9 = _mm256_permute2f128_pd(ymm12, ymm14, 0x20);
        ymm11 = _mm256_permute2f128_pd(ymm12, ymm14, 0x31);

        if(beta_val !=  0.0)
        {
            //1st col
            _mm256_storeu_pd((f_temp + 0), ymm0);
            _mm256_storeu_pd((f_temp + 4), ymm5);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 0))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            c0 += cs_c;

            //2nd col
            _mm256_storeu_pd((f_temp + 0), ymm1);
            for(i = 0; i < 4; i++)
                if(( m_off + i ) <= ( n_off + 1))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            for(; i < 6; i++)
                if((m_off + i ) <= ( n_off + 1))
                    c0[i] = c0[i] * beta_val + f_temp[i+2];
            c0 += cs_c;

            //3rd col
            _mm256_storeu_pd((f_temp + 0), ymm2);
            _mm256_storeu_pd((f_temp + 4), ymm7);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 2))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            c0 += cs_c;

            //4th col
            _mm256_storeu_pd((f_temp + 0), ymm3);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 3))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            for(; i < 6; i++)
                    if((m_off + i) <= ( n_off + 3))
                            c0[i] = c0[i] * beta_val + f_temp[i+2];
            c0 += cs_c;

            //5th col
            _mm256_storeu_pd((f_temp + 0), ymm4);
            _mm256_storeu_pd((f_temp + 4), ymm9);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 4))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            c0 += cs_c;

            //6th col
            _mm256_storeu_pd((f_temp + 0), ymm6);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 5))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            for(; i < 6; i++)
                if((m_off + i) <= ( n_off + 5))
                    c0[i] = c0[i] * beta_val + f_temp[i+2];
            c0 += cs_c;

            //7th col
            _mm256_storeu_pd((f_temp + 0), ymm8);
            _mm256_storeu_pd((f_temp + 4), ymm11);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 6))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            c0 += cs_c;

            //4th col
            _mm256_storeu_pd((f_temp + 0), ymm10);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 7))
                    c0[i] = c0[i] * beta_val + f_temp[i];
            for(; i < 6; i++)
                    if((m_off + i) <= ( n_off + 7))
                        c0[i] = c0[i] * beta_val + f_temp[i+2];

        }
        else
        {
             //1st col
            _mm256_storeu_pd((f_temp + 0), ymm0);
            _mm256_storeu_pd((f_temp + 4), ymm5);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 0))
                    c0[i] = f_temp[i];
            c0 += cs_c;

            //2nd col
            _mm256_storeu_pd((f_temp + 0), ymm1);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 1))
                    c0[i] = f_temp[i];
            for(; i < 6; i++)
                if((m_off + i) <= ( n_off + 1))
                    c0[i] = f_temp[i+2];
             c0 += cs_c;
 
            //3rd col
            _mm256_storeu_pd((f_temp + 0), ymm2);
            _mm256_storeu_pd((f_temp + 4), ymm7);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 2))
                    c0[i] = f_temp[i];
            c0 += cs_c;

            //4th col
            _mm256_storeu_pd((f_temp + 0), ymm3);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 3))
                    c0[i] = f_temp[i];
            for(; i < 6; i++)
                if((m_off + i) <= ( n_off + 3))
                    c0[i] = f_temp[i+2];
            c0 += cs_c;

            //5th col
            _mm256_storeu_pd((f_temp + 0), ymm4);
            _mm256_storeu_pd((f_temp + 4), ymm9);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 4))
                    c0[i] = f_temp[i];
            c0 += cs_c;

            //6th col
            _mm256_storeu_pd((f_temp + 0), ymm6);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 5))
                    c0[i] = f_temp[i];
            for(; i < 6; i++)
                if((m_off + i) <= ( n_off + 5))
                    c0[i] = f_temp[i+2];
            c0 += cs_c;

            //7th col
            _mm256_storeu_pd((f_temp + 0), ymm8);
            _mm256_storeu_pd((f_temp + 4), ymm11);
            for(i = 0; i < 6; i++)
                if(( m_off + i ) <= ( n_off + 6))
                    c0[i] = f_temp[i];
            c0 += cs_c;

            //4th col
            _mm256_storeu_pd((f_temp + 0), ymm10);
            for(i = 0; i < 4; i++)
                if(( m_off + i) <= ( n_off + 7))
                    c0[i] = f_temp[i];
            for(; i < 6; i++)
                if((m_off + i) <= ( n_off + 7))
                    c0[i] = f_temp[i+2];
       
        }

        return; 
    }
    else //General storage
    {
        if(beta_val != 0)
        {
            //Row 1
            _mm256_storeu_pd((f_temp + 0), ymm4);
            _mm256_storeu_pd((f_temp + 4), ymm5);

            for(j = 0; j < 8; j++)
                if(( m_off + 0 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

            //Row 2
            _mm256_storeu_pd((f_temp + 0), ymm6);
            _mm256_storeu_pd((f_temp + 4), ymm7);

            for(j = 0; j < 8; j++)
                if(( m_off + 1 )<= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

            //Row 3
            _mm256_storeu_pd((f_temp + 0), ymm8);
            _mm256_storeu_pd((f_temp + 4), ymm9);

            for(j = 0; j < 8; j++)
                if(( m_off + 2 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

            //Row 4
            _mm256_storeu_pd((f_temp + 0), ymm10);
            _mm256_storeu_pd((f_temp + 4), ymm11);

            for(j = 0; j < 8; j++)
                if(( m_off + 3 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

            //Row 5
            _mm256_storeu_pd((f_temp + 0), ymm12);
            _mm256_storeu_pd((f_temp + 4), ymm13);

            for(j = 0; j < 8; j++)
                if(( m_off + 4 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

            //Row 6
            _mm256_storeu_pd((f_temp + 0), ymm14);
            _mm256_storeu_pd((f_temp + 4), ymm15);

            for(j = 0; j < 8; j++)
                if(( m_off + 5 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = c0[ j * cs_c ] * beta_val + f_temp[j];
            c0 += rs_c;

        }
        else
        {
             //Row 1
            _mm256_storeu_pd((f_temp + 0), ymm4);
            _mm256_storeu_pd((f_temp + 4), ymm5);

            for(j = 0; j < 8; j++)
                if(( m_off + 0 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;

            //Row 2
            _mm256_storeu_pd((f_temp + 0), ymm6);
            _mm256_storeu_pd((f_temp + 4), ymm7);

            for(j = 0; j < 8; j++)
                if(( m_off + 1 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;

            //Row 3
            _mm256_storeu_pd((f_temp + 0), ymm8);
            _mm256_storeu_pd((f_temp + 4), ymm9);

            for(j = 0; j < 8; j++)
                if(( m_off + 2 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;

            //Row 4
            _mm256_storeu_pd((f_temp + 0), ymm10);
            _mm256_storeu_pd((f_temp + 4), ymm11);

            for(j = 0; j < 8; j++)
                if(( m_off + 3 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;

            //Row 5
            _mm256_storeu_pd((f_temp + 0), ymm12);
            _mm256_storeu_pd((f_temp + 4), ymm13);

            for(j = 0; j < 8; j++)
                if(( m_off + 4 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;

            //Row 6
            _mm256_storeu_pd((f_temp + 0), ymm14);
            _mm256_storeu_pd((f_temp + 4), ymm15);

            for(j = 0; j < 8; j++)
                if(( m_off + 5 ) <= ( n_off + j ))
                    c0[ j * cs_c ] = f_temp[j];
            c0 += rs_c;
       
        }

        return;
    }
}




