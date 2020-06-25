/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include <unistd.h>
#include "blis.h"

//#define FILE_IN_OUT
#define PRINT
#define MATRIX_INITIALISATION
int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t n, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   n_input, k_input;
	num_t dt;
	int   r, n_repeats;
	trans_t  transa;
	trans_t  transb;
	f77_char f77_transa;
	f77_char f77_transb;
	uplo_t uploc;
	f77_char f77_uploc;

	double dtime;
	double dtime_save;
	double gflops;
#ifdef FILE_IN_OUT
	FILE* fin  = NULL;
	FILE* fout = NULL;
	char gemm = 's';

#endif
	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

#ifndef PRINT
	p_begin = 48;
	p_end   = 10000;
	p_inc   = 192;

	n_input = -1;
	k_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	k_input = 50;
	n_input = 50;
#endif

#if 1
	//dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	transa = BLIS_NO_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;

	uploc  = BLIS_UPPER;

	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );
	bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );

#ifdef FILE_IN_OUT
	if (argc < 3)
	  {
	    printf("Usage: ./test_gemm_XX.x input.csv output.csv\n");
	    exit(1);
	  }
	fin = fopen(argv[1], "r");
	if (fin == NULL)
	  {
	    printf("Error opening the file %s\n", argv[1]);
	    exit(1);
	  }
	fout = fopen(argv[2], "w");
	if (fout == NULL)
	  {
	    printf("Error opening output file %s\n", argv[2]);
	    exit(1);
	  }
	fprintf(fout, "n\t k\t cs_a\t cs_b\t cs_c\t gflops\n");


	printf("~~~~~~~~~~_BLAS\t n\t k\t cs_a\t cs_b\t cs_c \t gflops\n");

	inc_t cs_a;
	inc_t cs_b;
	inc_t cs_c;

	while (fscanf(fin, "%ld %ld %ld %ld %ld %ld\n", &k, &n, &cs_a, &cs_b, &cs_c) == 5)
	  {
	    if ((m > cs_a) || (k > cs_b) || (m > cs_c)) continue; // leading dimension should be greater than number of rows
	    
	    bli_obj_create( dt, 1, 1, 0, 0, &alpha);
	    bli_obj_create( dt, 1, 1, 0, 0, &beta );

	    bli_obj_create( dt, n, k, 1, cs_a, &a );
	    bli_obj_create( dt, k, n, 1, cs_b, &b );
	    bli_obj_create( dt, n, n, 1, cs_c, &c );
	    bli_obj_create( dt, n, n, 1, cs_c, &c_save );
#ifdef MATRIX_INITIALISATION
	    bli_randm( &a );
	    bli_randm( &b );
	    bli_randm( &c );
#endif
	    bli_obj_set_struc( BLIS_TRINGULAR, &c );
	    bli_obj_set_uplo( uploc, &c );

	    bli_obj_set_conjtrans( transa, &a);
	    bli_obj_set_conjtrans( transb, &b);

	    //Randomize C and zero the unstored triangle to ensure the 
	    //implementation reads only from the stored region.
	    bli_randm( &c );
	    bli_mktrim( &c );

	    //bli_setsc( 0.0, -1, &alpha );
	    //bli_setsc( 0.0, 1, &beta );

	    bli_setsc( -1, 0.0, &alpha );
	    bli_setsc( 1, 0.0, &beta );

#else
	for ( p = p_begin; p <= p_end; p += p_inc )
	{
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		bli_obj_create( dt, n, k, k, 1, &a );
		bli_obj_create( dt, k, n, n, 1, &b );
		bli_obj_create( dt, n, n, n, 1, &c );
		bli_obj_create( dt, n, n, n, 1, &c_save );
#if 1
		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );
#else
		dim_t x, y, z;
		double* ap = (double *)bli_obj_buffer_at_off(&a);
		double* bp = (double *)bli_obj_buffer_at_off(&b);
		double* cp = (double *)bli_obj_buffer_at_off(&c);

		dim_t rs_a = bli_obj_row_stride(&a);
		dim_t rs_b = bli_obj_row_stride(&b);
		dim_t rs_c = bli_obj_row_stride(&c);

		FILE* fin = fopen("input.csv", "r");

		for(x = 0; x < n; x++)
			for(y = 0; y < k; y++)
				fscanf(fin,"%lf,", &ap[x*rs_a + y]);
		for(x = 0; x < k; x++)
			for(y = 0; y < n; y++)
				fscanf(fin, "%lf,", &bp[x*rs_b+y]);
		for(x = 0; x < n; x++)
			for(y= 0; y < n; y++)
				fscanf(fin, "%lf,", &cp[x*rs_c+y]);

		fclose(fin);
#endif
		bli_obj_set_struc( BLIS_TRIANGULAR, &c );
		bli_obj_set_uplo( uploc, &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

	        //Randomize C and zero the unstored triangle to ensure the 
	        //implementation reads only from the stored region.
#if 1
		bli_randm( &c );
	        bli_mktrim( &c );
#endif
		bli_setsc(  (1.0/1.0), 0.2, &alpha );
		bli_setsc(  (1.0/1.0), 0.3, &beta );

#endif
		bli_copym( &c, &c_save );
	
		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );


			dtime = bli_clock();


#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "," );
			bli_printm( "b", &b, "%4.1f", "," );
			bli_printm( "c", &c, "%4.1f", "," );
#endif

#ifdef BLIS

			bli_gemm( &alpha,
			          &a,
			          &b,
			          &beta,
			          &c );

#else

		if ( bli_is_float( dt ) )
		{
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			float*   alphap = bli_obj_buffer( &alpha );
			float*   ap     = bli_obj_buffer( &a );
			float*   bp     = bli_obj_buffer( &b );
			float*   betap  = bli_obj_buffer( &beta );
			float*   cp     = bli_obj_buffer( &c );
			
			sgemmt_( &f77_uploc,
				&f77_transa,
			        &f77_transb,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_double( dt ) )
		{
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_row_stride( &a );
			f77_int  ldb    = bli_obj_row_stride( &b );
			f77_int  ldc    = bli_obj_row_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  bp     = bli_obj_buffer( &b );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );
#if 1		
			cblas_dgemmt( 101, /* Row major */
				      121, /* lower triangular */
				      111, /* No transpose */
				      111, /* No transpose */
				      nn,
				      kk,
				      *alphap,
				      ap,lda,
				      bp, ldb,
				      *betap,
				      cp, ldc
				    );
#else
			dgemmt_(
				&f77_uploc,
				&f77_transa,
			        &f77_transb,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
#endif
		}
		else if ( bli_is_scomplex( dt ) )
		{
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			scomplex*  alphap = bli_obj_buffer( &alpha );
			scomplex*  ap     = bli_obj_buffer( &a );
			scomplex*  bp     = bli_obj_buffer( &b );
			scomplex*  betap  = bli_obj_buffer( &beta );
			scomplex*  cp     = bli_obj_buffer( &c );

			cgemmt_( &f77_uploc,
				&f77_transa,
			        &f77_transb,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_dcomplex( dt ) )
		{
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			dcomplex*  alphap = bli_obj_buffer( &alpha );
			dcomplex*  ap     = bli_obj_buffer( &a );
			dcomplex*  bp     = bli_obj_buffer( &b );
			dcomplex*  betap  = bli_obj_buffer( &beta );
			dcomplex*  cp     = bli_obj_buffer( &c );

			zgemmt_( &f77_uploc,
				&f77_transa,
			        &f77_transb,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * n * k * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_gemm_blis" );
#else
		printf( "data_gemm_%s", BLAS );
#endif


#ifdef FILE_IN_OUT
		printf("%4lu \t %4lu \t %4lu \t %4lu \t %4lu \t %6.3f\n", \
		        ( unsigned long )n,
		       ( unsigned long )k, (unsigned long)cs_a, (unsigned long)cs_b, (unsigned long)cs_c,  gflops );
		
		
		fprintf(fout, "%4lu \t %4lu \t %4lu \t %4lu \t %4lu \t %6.3f\n", \
		        ( unsigned long )n,
		        ( unsigned long )k, (unsigned long)cs_a, (unsigned long)cs_b, (unsigned long)cs_c,  gflops );
		fflush(fout);

#else
		printf( "( %2lu, 1:4 ) = [ %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )n,
		        ( unsigned long )k, gflops );
#endif
		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	//bli_finalize();
#ifdef FILE_IN_OUT
    fclose(fin);
    fclose(fout);
#endif
	return 0;
}

