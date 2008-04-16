MODULE hmm_for
  IMPLICIT NONE
    CONTAINS

      SUBROUTINE ALPHA_SCALED(A, B, PI, R, S, N, T)
        INTEGER :: N, T
        DOUBLE PRECISION, INTENT(IN), DIMENSION(N,N) :: A
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: B
        DOUBLE PRECISION, INTENT(IN), DIMENSION(N) :: PI
        DOUBLE PRECISION, INTENT(OUT), DIMENSION(T,N) :: R ! alpha_scaled
        DOUBLE PRECISION, INTENT(OUT), DIMENSION(T) :: S   ! scaling_factors
        INTEGER :: I, J

        !PRINT *, "ALPHA_SCALED N=", N, "T=", T

        ! Compute alpha( 1, i )
        S = 0.0
        
        DO J=1, N
           R(1,J) = B(1,J) * PI(J)
           S(1) = S(1) + R(1,J)
        END DO
        S(1) = 1./S(1)

        ! Normalize
        R(1,:) = R(1,:) * S(1)

        DO I=2, T
           S(I) = 0.
           DO J=1, N
              R(I,J) = DOT_PRODUCT(A(:,J),R(I-1,:))*B(I,J)
              S(I) = S(I) + R(I,J)
           END DO
           S(I) = 1. / S(I)
           R(I,:) = R(I,:) * S(I)
        END DO
      END SUBROUTINE ALPHA_SCALED


      SUBROUTINE BETA_SCALED( A, B, R, S, N, T )
        INTEGER :: N, T
        DOUBLE PRECISION, INTENT(IN), DIMENSION(N,N) :: A
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: B
        DOUBLE PRECISION, INTENT(OUT), DIMENSION(T,N) :: R
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T) :: S
        DOUBLE PRECISION, DIMENSION(N) :: tmp
        INTEGER :: I
        
        R(T,1:N) = S(T)

        DO I=T-1,1,-1
           tmp = B(I+1,1:N) * S(I) * R(I+1,1:N)
           R(I,1:N) = MATMUL( A, tmp )
        END DO
      END SUBROUTINE BETA_SCALED

      SUBROUTINE HMM_KSI( A, B, AL, BE, KSI, N, T )
        INTEGER :: N, T
        DOUBLE PRECISION, INTENT(IN), DIMENSION(N,N) :: A
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: B
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: AL
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: BE
        DOUBLE PRECISION, INTENT(OUT), DIMENSION(T-1,N,N) :: KSI
        INTEGER :: I, J, K
        DOUBLE PRECISION :: SUM

!        PRINT *, "HMM_KSI N=", N, "T=", T

        DO I=1,T-1
           SUM = 0.0
           DO J=1,N
              DO K=1,N
                 KSI(I,J,K) = A(J,K)*B(I+1,K)*BE(I+1,K)*AL(I,J)
                 SUM = SUM + KSI(I,J,K)
              END DO
           END DO
           IF (SUM .NE. 0.0) THEN
              SUM = 1./SUM
              KSI(I,1:N,1:N) = KSI(I,1:N,1:N) * SUM
           END IF
        END DO

      END SUBROUTINE HMM_KSI


      SUBROUTINE UPDATE_ITER_B( G, OBS, B_bar, M, N, T )
        INTEGER :: M, N, T
        DOUBLE PRECISION, INTENT(IN), DIMENSION(T,N) :: G
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(M,N) :: B_bar
        INTEGER, INTENT(IN), DIMENSION(T) :: OBS
        INTEGER :: I

!        PRINT *, "UPDATE_ITER_B N=", N, "T=", T, "M=", M

        DO I=1, T
           B_bar(OBS(I)+1, : ) = B_bar( OBS(I)+1, :) + G( I, : )
        END DO

      END SUBROUTINE UPDATE_ITER_B

      SUBROUTINE CORRECTM( G, IDX, V, M, N )
        INTEGER :: M, N, I, IDX
        DOUBLE PRECISION :: V
        DOUBLE PRECISION :: S
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(M,N) :: G

!        PRINT *, "CORRECTM N=", N, "M=", M

        IF (IDX .EQ. 1) THEN
           DO I=1,M
              S = SUM(G(I,:))
              IF (S .EQ. 0.0) G(I,:) = V
           END DO
        ELSE
           DO I=1,N
              S = SUM(G(:,I))
              IF (S .EQ. 0.0) G(:,I) = V
           END DO           
        END IF
      END SUBROUTINE CORRECTM

      SUBROUTINE NORMALIZE_B( B, V, M, N)
        INTEGER :: M, N, I, J
        DOUBLE PRECISION :: W
        DOUBLE PRECISION, INTENT(INOUT), DIMENSION(M,N) :: B
        DOUBLE PRECISION, INTENT(IN), DIMENSION(N) :: V
!        PRINT *, "NORMALIZE_B N=", N, "M=", M

        DO J=1,N
           IF (V(J) .NE. 0.0) THEN
              W = 1./V(J)
           ELSE
              W = 1.0
           END IF
           DO I=1,M
              B(I,J) = B(I,J)*W
           END DO
        END DO
      END SUBROUTINE NORMALIZE_B

      SUBROUTINE GAMMA( AL, B, S, G, M, N )
        DOUBLE PRECISION, INTENT(IN), DIMENSION(M,N) :: B
        DOUBLE PRECISION, INTENT(IN), DIMENSION(M,N) :: AL
        DOUBLE PRECISION, INTENT(IN), DIMENSION(M) :: S   ! scaling_factors
        DOUBLE PRECISION, INTENT(OUT), DIMENSION(M,N) :: G
        DOUBLE PRECISION, DIMENSION(M) :: SI
        INTEGER :: T, M, N

        SI(:) = 1./S(:)
        DO T=1,M
           G(T,:) = AL(T,:)*B(T,:)/S(T)
        END DO
      END SUBROUTINE GAMMA
    END MODULE hmm_for

