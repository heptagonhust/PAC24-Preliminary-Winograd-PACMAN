#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include "kblas.h"


const float A_T[6][8] = {
    {1,  1,  1,  1,  1,  1,  1,  0},
    {0,  1, -1,  2, -2, 1/2.0, -1/2.0, 0},
    {0,  1,  1,  4,  4, 1/4.0,  1/4.0, 0},
    {0,  1, -1,  8, -8, 1/8.0, -1/8.0, 0},
    {0,  1,  1, 16, 16, 1/16.0, 1/16.0, 0},
    {0,  1, -1, 32, -32, 1/32.0, -1/32.0, 1},
};

const float A[8][6] = {
    {1, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1},
    {1, -1, 1, -1, 1, -1},
    {1, 2, 4, 8, 16, 32},
    {1, -2, 4, -8, 16, -32},
    {1, 1/2.0, 1/4.0, 1/8.0, 1/16.0, 1/32.0},
    {1, -1/2.0, 1/4.0, -1/8.0, 1/16.0, -1/32.0},
    {0, 0, 0, 0, 0, 1}
};



const float G[8][3] = {
    {1,    0,    0},
    {-2/9.0, -2/9.0, -2/9.0},
    {-2/9.0,  2/9.0, -2/9.0},
    {1/90.0,  1/45.0,  2/45.0},
    {1/90.0, -1/45.0,  2/45.0},
    {32/45.0,  16/45.0,  8/45.0},
    {32/45.0, -16/45.0,  8/45.0},
    {0,    0,    1}
};

const float G_T[3][8] = {
    {1, -2/9.0, -2/9.0, 1/90.0, 1/90.0, 32/45.0, 32/45.0, 0},
    {0, -2/9.0, 2/9.0, 1/45.0, -1/45.0, 16/45.0, -16/45.0, 0},
    {0, -2/9.0, -2/9.0, 2/45.0, 2/45.0, 8/45.0, 8/45.0, 1}
};


const float B_T[8][8] = {
    {1,   0, -21/4.0, 0,  21/4.0,  0, -1,  0},
    {0,   1,    1,  -17/4.0,  -17/4.0,  1,  1,  0},
    {0,  -1,    1,   17/4.0, -17/4.0, -1,  1,  0},
    {0,  1/2.0, 1/4.0, -5/2.0, -5/4.0,  2,  1,  0},
    {0, -1/2.0, 1/4.0,  5/2.0, -5/4.0, -2,  1,  0},
    {0,     2,     4,  -5/2.0,   -5,  1/2.0,  1,  0},
    {0,    -2,     4,  5/2.0,   -5, -1/2.0,  1,  0},
    {0,   -1,  0, 21/4.0, 0, -21/4.0, 0,  1}
};

const float B[8][8] = {
    {1, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, -1, 1/2.0, -1/2.0, 2, -2, -1},
    {-21/4.0, 1, 1, 1/4.0, 1/4.0, 4, 4, 0},
    {0, -17/4.0, 17/4.0, -5/2.0, 5/2.0, -5/2.0, 5/2.0, 21/4.0},
    {21/4.0, -17/4.0, -17/4.0, -5/4.0, -5/4.0, -5, -5, 0},
    {0, 1, -1, 2, -2, 1/2.0, -1/2.0, -21/4.0},
    {-1, 1, 1, 1, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 1}
};




void gemm8x8( const float *A, const float *B,float *C) {
                svbool_t pg = svwhilelt_b32(0, 8);
                //svfloat32_t c = svdup_f32(0.0f);
                svfloat32_t a;
                svfloat32_t b0 = svld1(pg, B);
                svfloat32_t b1 = svld1(pg, B + 8);
                svfloat32_t b2 = svld1(pg, B + 16);
                svfloat32_t b3 = svld1(pg, B + 24);  
                svfloat32_t b4 = svld1(pg, B + 32);  
                svfloat32_t b5 = svld1(pg, B + 40);  
                svfloat32_t b6 = svld1(pg, B + 48);  
                svfloat32_t b7 = svld1(pg, B + 56);  

                svfloat32_t c0;// = svdup_f16(0.0f);
                svfloat32_t c1;// = svdup_f16(0.0f);
                svfloat32_t c2 ;//= svdup_f16(0.0f);
                svfloat32_t c3 ;//= svdup_f16(0.0f);
                svfloat32_t c4 ;//= svdup_f16(0.0f);
                svfloat32_t c5 ;//= svdup_f16(0.0f);
                svfloat32_t c6 ;//= svdup_f16(0.0f);
                svfloat32_t c7 ;//= svdup_f16(0.0f);
                //float sum = 0.0f;
                const int N = 8;
    for (int i = 0; i < N; ++i) {
         
            // Initialize the result element
               
                a = svld1(pg, A + i * 8);
                // Perform the multiply-add
               // c = svmla_f32_m(pg, c, a, b);
                c0 = svmul_f32_x(pg, a, b0);
                c1 = svmul_f32_x(pg, a, b1);
                c2 = svmul_f32_x(pg, a, b2);
                c3 = svmul_f32_x(pg, a, b3);
                c4 = svmul_f32_x(pg, a, b4);
                c5 = svmul_f32_x(pg, a, b5);
                c6 = svmul_f32_x(pg, a, b6);
                c7 = svmul_f32_x(pg, a, b7);
            

            // Sum all partial results within the vector register
                C[i * N ] = svaddv_f32(pg, c0);
              C[i * N + 1] = svaddv_f32(pg, c1);
              C[i * N + 2] = svaddv_f32(pg, c2);
              C[i * N + 3] = svaddv_f32(pg, c3);
              C[i * N + 4] = svaddv_f32(pg, c4);
              C[i * N + 5] = svaddv_f32(pg, c5);
              C[i * N + 6] = svaddv_f32(pg, c6);
              C[i * N + 7] = svaddv_f32(pg, c7);

            // Store the result
             
        
    }   
}





void sgemm(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < M * N; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
       for (int k = 0; k < K; ++k)
          out[i * N + j]  += A[i * K + k] * B[k * N + j];
}

void sgemm_parallel(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < M * N; ++i) {
    out[i] = 0.0f;
  }

for (int k = 0; k < K; ++k)
#pragma omp parallel for collapse(2)
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
          out[(long)i * N + j]  += A[i * K + k] * B[k * N + j];
    }
}
// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {
  // m = 2; r = 3; alpha = 4
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const long sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;         //! 正确的

  int h_remainder = outHeight % 6;        // ! 多余的部分
  int w_remainder = outWidth % 6;

  int TH = outHeight / 6 ;
  int TW = outWidth / 6;
  if(h_remainder) TH = outHeight / 6 + 1;
  if(w_remainder) TW = outWidth / 6 + 1;
  const long P = TH * TW * N;        // !   得到总共的 tile 数量 这个是一个image的


  float tmp_u[24];  // 4 * 3 //!
  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
#pragma omp parallel for collapse(2) private(tmp_u) schedule(static)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float *filters_ptr = filter + (k * C + c) * sizeF;
       cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 8, 3, 3, 1.0, &G[0][0], 3, filters_ptr, 3, 0, tmp_u, 3);
       cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 8, 8, 3, 1.0, tmp_u, 3, &G_T[0][0], 8, 0, U+(k * C + c)* 64, 8);
      
    }
  }




  // V[:, :, c, p] = B_T * image[c, b, :, :] * B
  float tmp_v[64];
   float d[64]; // d: [4 * 4];
  float v[64];  // v: [4 * 4];
#pragma omp parallel for collapse(2) private(tmp_v, d, v) schedule(static) num_threads(160)
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
    // ! 处理列的特殊情况 y
    
      int y =0; 
      for ( y = 0; y < outHeight / 6; ++y) {
        int x = 0;      // ! 特殊情况下使用 x
        for ( x = 0; x < outWidth / 6; ++x) {

          // Generate d_cb
           #pragma clang loop vectorize(enable)
          #pragma clang loop interleave(enable)
          for (int iy = 0; iy < 8; ++iy)
            for (int ix = 0; ix < 8; ++ix)
              d[ix * 8 + iy] = image[(n * C + c) * sizeI +
                                     (y * 6 + iy) * inWidth + (x * 6 + ix)];
          // sgemm(&B_T[0][0], d, tmp_v, 8, 8, 8);
          // sgemm(tmp_v, &B[0][0], v, 8, 8, 8);
          int b = (n * TH + y) * TW + x; 
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, &B_T[0][0], 8, d, 8, 0, tmp_v, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, tmp_v, 8, &B_T[0][0], 8, 0, V+(c * P + b) * 64, 8);          //gemm8x8(&B_T[0][0], d, tmp_v);
          //  gemm8x8(tmp_v, &B_T[0][0], v);

          //int b = (n * TH + y) * TW + x;           // ! - > TH TW
         //  #pragma clang loop vectorize(enable)
         // #pragma clang loop interleave(enable)
          //for (int xi = 0; xi < 8; ++xi)
            //for (int nu = 0; nu < 8; ++nu)
              //V[(c * P + b) * 64  + xi * 8 + nu] = v[xi * 8 + nu];
        }
        // ! 处理这个边界情况 
        if(w_remainder) {
            memset(d, 0, 256);           //! 先完全置0  后面只用置原来image的多余的值
            for (int iy = 0; iy < 8; ++iy)    // ! 还是需要8行 但是不需要 8列
              for (int ix = 0; ix < (2 + w_remainder); ++ix) 
                  d[ix * 8 + iy] = image[(n * C + c) * sizeI +
                                      (y * 6 + iy) * inWidth +( x * 6 + ix)];   //! x的值需要改变
                  // sgemm(&B_T[0][0], d, tmp_v, 8, 8, 8);
                  // sgemm(tmp_v, &B[0][0], v, 8, 8, 8);
                  //  gemm8x8(&B_T[0][0], d, tmp_v);
                  // gemm8x8(tmp_v, &B_T[0][0], v);
                  int b = (n * TH + y) * TW + x;
                  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, &B_T[0][0], 8, d, 8, 0, tmp_v, 8);
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, tmp_v, 8, &B_T[0][0], 8, 0, V+(c * P + b) * 64 , 8); 
              //    int b = ((n * TH) + y) * TW + x;        // ! x is same okay
            //       #pragma clang loop vectorize(enable)
          //#pragma clang loop interleave(enable)
                  //for (int xi = 0; xi < 8; ++xi)
                   // for (int nu = 0; nu < 8; ++nu)
                     // V[(c * P + b) * 64  + xi * 8 + nu] = v[xi * 8 + nu];     // ! V[ c, : p, :, : ]
            
        }
      }
    // ! colume _ remainder 
    if(h_remainder) {       // ! 最后还要在内部判断 一个 w_remainer 来处理最右下角的这个特殊块
      int x =0;
        for( x = 0; x < outWidth / 6; ++x){
            memset(d, 0, 256);  // ! all zero 0 
            for (int iy = 0; iy < (2 + h_remainder); ++iy)            // ! iy is more
                for (int ix = 0; ix < 8; ++ix)
                d[ix * 8 + iy] = image[(n * C + c) * sizeI +
                                        (y * 6 + iy) * inWidth + (x * 6 + ix)];
            // sgemm(&B_T[0][0], d, tmp_v, 8, 8, 8);
            // sgemm(tmp_v, &B[0][0], v, 8, 8, 8);
          //    gemm8x8(&B_T[0][0], d, tmp_v);
          //  gemm8x8(tmp_v, &B_T[0][0], v);
          int b = (n * TH + y) * TW + x;
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, &B_T[0][0], 8, d, 8, 0, tmp_v, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, tmp_v, 8, &B_T[0][0], 8, 0, V+(c * P + b) * 64, 8); 
            //int b = (n * TH + y) * TW + x;           // ! - > TH TW    y 是对的
             //#pragma clang loop vectorize(enable)
            //#pragma clang loop interleave(enable)
            //for (int xi = 0; xi < 8; ++xi)
              //  for (int nu = 0; nu < 8; ++nu)
                //V[(c * P + b) * 64  + xi * 8 + nu] = v[xi * 8 + nu];
        }
        if(w_remainder){                    //! 处理最右下角的tile
            memset(d, 0, 256);  // ! all zero 0 
            for (int iy = 0; iy < (2 + h_remainder); ++iy)            // ! iy is more
                for (int ix = 0; ix < (2 + w_remainder); ++ix)
                d[ix * 8 + iy] = image[(n * C + c) * sizeI +
                                        (y * 6 + iy) * inWidth + (x * 6 + ix)];
            // sgemm(&B_T[0][0], d, tmp_v, 8, 8, 8);
            // sgemm(tmp_v, &B[0][0], v, 8, 8, 8);
          //    gemm8x8(&B_T[0][0], d, tmp_v);
          //  gemm8x8(tmp_v, &B_T[0][0], v);
          int b = (n * TH + y) * TW + x;
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, &B_T[0][0], 8, d, 8, 0, tmp_v, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans, 8, 8, 8, 1.0, tmp_v, 8, &B_T[0][0], 8, 0, V+(c * P + b) * 64, 8); 
            //int b = (n * TH + y) * TW + x;           // ! - > TH TW    y 是对的
             //#pragma clang loop vectorize(enable)
            //#pragma clang loop interleave(enable)
           // for (int xi = 0; xi < 8; ++xi)
             //   for (int nu = 0; nu < 8; ++nu)
               // V[(c * P + b) * 64  + xi * 8 + nu] = v[xi * 8 + nu];
        }
    }
    }
    
    


  // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
  // for (int xi = 0; xi < 8; ++xi) {
  //   for (int nu = 0; nu < 8; ++nu) {
  //     float *M_ptr = M + (long)(xi * 8 + nu) * K * P;
  //     float *U_ptr = U + (long)(xi * 8 + nu) * K * C;
  //     float *V_ptr = V + (long)(xi * 8 + nu) * C * P;
  //     sgemm_parallel(U_ptr, V_ptr, M_ptr, K, C, P);
  //   }
  // }
  
  
  

// ! 似乎不需要改动'
/*
#pragma omp parallel for collapse(2)
 for(int k=0;k<K;k++ ){
    for(int p=0;p<P;p++){
      float *M_ptr = M + (long)(k*P+p)*64;
      // for(int l=0;l<64;l++){
      //   M_ptr[l] = 0;
      // }
      for(int c=0;c<C;c++){
        
        float *U_ptr = U + (long)(k*C+c)*64;
        float *V_ptr = V + (long)(c*P+p)*64;
        //#pragma omp simd
       #pragma clang loop vectorize(enable)
        #pragma clang loop interleave(enable)
        for(int i=0;i<8;i++){
          for(int j=0;j<8;j++){
            M_ptr[i*8+j] += U_ptr[i*8+j]*V_ptr[i*8+j]; 
          }
        }
      }

    }
  }
*/

#pragma omp parallel for collapse(2) schedule(static) num_threads(160)
 for(int k=0;k<K;k++ ){
    for(int p=0;p<P;p++){
      float *M_ptr = M + (long)(k*P+p)*64;
      // for(int l=0;l<64;l++){
      //   M_ptr[l] = 0;
      // }
      for(int c=0;c<C;c++){
        
        float *U_ptr = U + (long)(k*C+c)*64;
        //prefetch(U+(long)(k*C+c)*64);
        //prefetch(V+(long)(k*C+c)*64);
        float *V_ptr = V + (long)(c*P+p)*64;
        for(int i=0;i<8;i++){
          //__builtin_prefetch(U_ptr+8*i,0,1);
         // __builtin_prefetch(V_ptr+8*i,0,1);
          //__builtin_prefetch(M_ptr+8*i,1,1);
          float32x4_t udata0 = vld1q_f32(U_ptr+8*i);
          float32x4_t udata1 = vld1q_f32(U_ptr+8*i+ 4);
          float32x4_t vdata0 = vld1q_f32(V_ptr+8*i);
          float32x4_t vdata1 = vld1q_f32(V_ptr+8*i+ 4);
          float32x4_t odata0 = vld1q_f32(M_ptr+8*i);
          float32x4_t odata1 = vld1q_f32(M_ptr+8*i+ 4);
          float32x4_t outData0 = vmulq_f32(udata0, vdata0);
          float32x4_t outData1 = vmulq_f32(udata1, vdata1);
          vst1q_f32(M_ptr+8*i, vaddq_f32(odata0, outData0));
          vst1q_f32(M_ptr+8*i+ 4, vaddq_f32(odata1, outData1));
        }

      }

    }
  }



// #pragma omp parallel for collapse(2)
//  for(int k=0;k<K;k++ ){
//     for(int p=0;p<P;p++){
//       float *M_ptr = M + (long)(k*P+p)*64;          // ! 找到这个tile
//     svbool_t pg = svwhilelt_b32(0, 8);
//     svfloat32_t m0 = svld1(pg, M_ptr + 0);
//     svfloat32_t m1 =svld1(pg, M_ptr + 8);
//     svfloat32_t m2 =svld1(pg, M_ptr + 16);
//     svfloat32_t m3 =svld1(pg, M_ptr + 24);
//     svfloat32_t m4 =svld1(pg, M_ptr + 32);
//     svfloat32_t m5 =svld1(pg, M_ptr + 40);
//     svfloat32_t m6 =svld1(pg, M_ptr + 48);
//     svfloat32_t m7 =svld1(pg, M_ptr + 56);

//     svfloat32_t u0 ;
//     svfloat32_t v0 ;
//       for(int c=0;c<C;c++){
        
//         float *U_ptr = U + (long)(k*C+c)*64;
//         float *V_ptr = V + (long)(c*P+p)*64;
   
//             u0 = svld1(pg, U_ptr + 0);
//             v0 = svld1(pg, V_ptr + 0);
//             m0 = svmla_f32_m(pg, m0, u0, v0);
//            // M_ptr[i*8+j] += U_ptr[i*8+j]*V_ptr[i*8+j]; 
//             u0 = svld1(pg, U_ptr + 8);
//             v0 = svld1(pg, V_ptr + 8);
//             m1 = svmla_f32_m(pg, m1, u0, v0);
            
//              u0 = svld1(pg, U_ptr + 16);
//             v0 = svld1(pg, V_ptr + 16);
//             m2 = svmla_f32_m(pg, m2, u0, v0);

//              u0 = svld1(pg, U_ptr + 24);
//             v0 = svld1(pg, V_ptr + 24);
//             m3 = svmla_f32_m(pg, m3, u0, v0);

//              u0 = svld1(pg, U_ptr + 32);
//             v0 = svld1(pg, V_ptr + 32);
//             m4 = svmla_f32_m(pg, m4, u0, v0);

//              u0 = svld1(pg, U_ptr + 40);
//             v0 = svld1(pg, V_ptr + 40);
//             m5 = svmla_f32_m(pg, m5, u0, v0);

//              u0 = svld1(pg, U_ptr + 48);
//             v0 = svld1(pg, V_ptr + 48);
//             m6 = svmla_f32_m(pg, m6, u0, v0);

//              u0 = svld1(pg, U_ptr + 56);
//             v0 = svld1(pg, V_ptr + 56);
//             m7 = svmla_f32_m(pg, m7, u0, v0);

           
//           //}
//         // }
//       }
//        svst1(pg, M_ptr + 0,m0);
//        svst1(pg, M_ptr + 8,m1);
//        svst1(pg, M_ptr + 16,m2);
//        svst1(pg, M_ptr + 24,m3);
//        svst1(pg, M_ptr + 32,m4);
//        svst1(pg, M_ptr + 40,m5);
//        svst1(pg, M_ptr + 48,m6);
//        svst1(pg, M_ptr + 56,m7);

//     }
//   }

  // Y = A_T * m * A
  float mm[64];       // 4 * 4
  float tmp_m[48];     // 2 * 4
  float temp_out[36];  // 2 * 2
#pragma omp parallel for collapse(2) private(mm, temp_out, tmp_m) schedule(static) num_threads(160)
  for (int n = 0; n < N; ++n)               // ! 对于每一个image
    for (int k = 0; k < K; ++k) {
        int y = 0;          // ! col
      for ( y = 0; y < outHeight / 6; ++y) {
        int x = 0;          // ! row
        for ( x = 0; x < outWidth / 6; ++x) {          // ! -> TH TW
          int b = (n * TH + y) * TW + x;
           #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)                        // ! 确定这个是第几个tile
          for (long xi = 0; xi < 8; ++xi) {
            for (long nu = 0; nu < 8; ++nu) {
              //mm[xi * 8 + nu] = M[((xi * 8 + nu) * K + k) * P + b];
              mm[xi * 8 + nu] = M[(k*P+b)*64+8*xi+nu];              // ! 读取kernel
            }
          }
          //sgemm(&A_T[0][0], mm, tmp_m, 6, 8, 8);
          //sgemm(tmp_m, &A[0][0], temp_out, 6, 8, 6);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 8, 8, 1.0, &A_T[0][0], 8, mm, 8, 0.0, tmp_m, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 6, 8, 1.0,tmp_m, 8, &A[0][0], 6, 0.0, temp_out, 6);

          #pragma clang loop vectorize(enable)
        #pragma clang loop interleave(enable)
          for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
              out[(long)((n * K + k) * outHeight + y * 6 + i) * outWidth + x * 6 +
                  j] = temp_out[i * 6 + j];
        }
        if(w_remainder){
            int b = (n * TH + y) * TW + x; 
             #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)                       // ! 确定这个是第几个tile
                for (long xi = 0; xi < 8; ++xi) {
                    for (long nu = 0; nu < 8; ++nu) {
                      
                    //mm[xi * 8 + nu] = M[((xi * 8 + nu) * K + k) * P + b];
                    mm[xi * 8 + nu] = M[(k*P+b)*64+8*xi+nu];              // ! 读取kernel
                    }
                }
          // sgemm(&A_T[0][0], mm, tmp_m, 6, 8, 8);
          // sgemm(tmp_m, &A[0][0], temp_out, 6, 8, 6);
           cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 8, 8, 1.0, &A_T[0][0], 8, mm, 8, 0.0, tmp_m, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 6, 8, 1.0,tmp_m, 8, &A[0][0], 6, 0.0, temp_out, 6);
            #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)
          for (int i = 0; i < 6; ++i)
            for (int j = 0; j < w_remainder; ++j)
         
              out[(long)((n * K + k) * outHeight + y * 6 + i) * outWidth + x * 6 +
                  j] = temp_out[i * 6 + j];
        }
      }
      if(h_remainder){
            int x =0;
            for( x = 0;x < outWidth / 6; ++x){
                int b = (n * TH + y) * TW + x; 
                  #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)                       // ! 确定这个是第几个tile
                for (long xi = 0; xi < 8; ++xi) {
                    for (long nu = 0; nu < 8; ++nu) {
                     
                    //mm[xi * 8 + nu] = M[((xi * 8 + nu) * K + k) * P + b];
                    mm[xi * 8 + nu] = M[(k*P+b)*64+8*xi+nu];              // ! 读取kernel
                    }
                }
            // sgemm(&A_T[0][0], mm, tmp_m, 6, 8, 8);
            // sgemm(tmp_m, &A[0][0], temp_out, 6, 8, 6);
             cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 8, 8, 1.0, &A_T[0][0], 8, mm, 8, 0.0, tmp_m, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 6, 8, 1.0,tmp_m, 8, &A[0][0], 6, 0.0, temp_out, 6);
            #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)
            for (int i = 0; i < h_remainder; ++i)
                for (int j = 0; j < 6; ++j)
                out[(long)((n * K + k) * outHeight + y * 6 + i) * outWidth + x * 6 +
                    j] = temp_out[i * 6 + j];
            }
            if(w_remainder){
                int b = (n * TH + y) * TW + x;  
                 #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)                      // ! 确定这个是第几个tile
                for (long xi = 0; xi < 8; ++xi) {
                    for (long nu = 0; nu < 8; ++nu) {
                    //mm[xi * 8 + nu] = M[((xi * 8 + nu) * K + k) * P + b];
                    mm[xi * 8 + nu] = M[(k*P+b)*64+8*xi+nu];              // ! 读取kernel
                    }
                }
          // sgemm(&A_T[0][0], mm, tmp_m, 6, 8, 8);
          // sgemm(tmp_m, &A[0][0], temp_out, 6, 8, 6);
           cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 8, 8, 1.0, &A_T[0][0], 8, mm, 8, 0.0, tmp_m, 8);
          cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, 6, 6, 8, 1.0,tmp_m, 8, &A[0][0], 6, 0.0, temp_out, 6);
           #pragma clang loop vectorize(enable)
            #pragma clang loop interleave(enable)
          for (int i = 0; i < h_remainder; ++i)
            for (int j = 0; j < w_remainder; ++j)
              out[(long)((n * K + k) * outHeight + y * 6 + i) * outWidth + x * 6 +
                  j] = temp_out[i * 6 + j];
            }
      }
    }
    
}