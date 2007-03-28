/*
  Copyright (c) 2002 LOGILAB S.A. (Paris, FRANCE).
  http://www.logilab.fr/ -- mailto:contact@logilab.fr
 
  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.
 
  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc.,
  59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*/

#include <Python.h>
#include <stdio.h>
#include <stdarg.h>
#include "numpy/arrayobject.h"

#define STRIDE(A,n) (A->strides[n]/sizeof(double))
#define STRIDE_LONG(A,n) (A->strides[n]/sizeof(long))

#ifdef B0
/* On MacOS at least, termios.h is included by Python.h and defines B0 to be 0 */
#undef B0
#endif

DL_EXPORT(void) init_hmm_c(void);

const char* __revision__ = "$Id: _hmm.c,v 1.5 2003-10-13 15:41:04 ludal Exp $";

/* --------------------------------------------------------------
   Helper functions for HMM models
   check_A returns a PyArrayObject or NULL and N such that shape(A)=(N,N)
   check_B idem, make sure that shape(B)=(T,N) update T
   check_PI idem checks shape(PI)==N
   All three functions raise exceptions when they return NULL
   --------------------------------------------------------------
*/

/* Formats an error string and throw a ValueError exception with the string */
static void throwexcept( char* msg, ... )
{
    char errstring[1000];
    va_list ap;
    va_start(ap, msg);
    vsnprintf(errstring,1000,msg,ap);
    PyErr_SetString(PyExc_ValueError,errstring);
}

/* Checks if oA is can be a square matrix of doubles.
 * Returns a PyArrayObject built from oA, and its size in N
 */
static PyArrayObject* check_square( PyObject* oA, long* N, char* name )
{
    PyArrayObject *A;
    A = (PyArrayObject*)PyArray_ContiguousFromObject(oA,PyArray_DOUBLE,2,2);
    if (A==NULL) {
	/*throwexcept("Array %s must be two-dimensionnal",name);*/
	goto err;
    }
    if (A->dimensions[0]!=A->dimensions[1]) {
	throwexcept("Array %s must be square, it is %dx%d", name, A->dimensions[0],
		    A->dimensions[1]);
	goto err;
    }
    if (A->dimensions[0]==0) {
	throwexcept("Array %s has size 0",name);
	goto err;
    }
    *N = A->dimensions[0];
    return A;
 err:
    if (A) {
	Py_DECREF(A);
    }
    return NULL;
}

static PyArrayObject* check_second_dim( PyObject* oB, long *T, long N, char* name )
{ 
    PyArrayObject *B;
    B = (PyArrayObject*)PyArray_ContiguousFromObject(oB,PyArray_DOUBLE,2,2);
    if (B==NULL || B->dimensions[1]!=N
	|| B->dimensions[1]==0 || B->dimensions[0]==0) {
	throwexcept("Array %s, must be two-dimensionnal, of type float, and "\
		    "have matching dimension with A",name);
	goto err;
    }
    *T = B->dimensions[0];
    return B;
 err:
    if (B) {
	Py_DECREF(B);
    }
    return NULL;
}

static PyArrayObject* check_2D( PyObject* oB, long *M, long *N, char* name )
{ 
    PyArrayObject *B;
    B = (PyArrayObject*)PyArray_ContiguousFromObject(oB,PyArray_DOUBLE,2,2);
    if (B==NULL||B->dimensions[0]==0||B->dimensions[1]==0) {
	throwexcept("Array %s must be two-dimensionnal and of type Float", name);
	goto err;
    }
    *M = B->dimensions[0];
    *N = B->dimensions[1];
    return B;
 err:
    if (B) {
	Py_DECREF(B);
    }
    return NULL;
}

static PyArrayObject* check_one_dim( PyObject* oPI, long N, char* name )
{
    PyArrayObject *PI;
    PI = (PyArrayObject*)PyArray_ContiguousFromObject(oPI,PyArray_DOUBLE,1,1);
    if (PI==NULL || PI->dimensions[0]!=N) {
	throwexcept("Array %s, must be one-dimensionnal, of type float, and " \
		    "have matching dimension with A",name);
	goto err;
    }
    return PI;
 err:
    if (PI) {
	Py_DECREF(PI);
    }
    return NULL;
}

static PyArrayObject* check_long_one_dim( PyObject* oPI, long N, char* name )
{
    PyArrayObject *PI;
    PI = (PyArrayObject*)PyArray_ContiguousFromObject(oPI,PyArray_LONG,1,1);
    if (PI==NULL || PI->dimensions[0]!=N) {
	throwexcept("Array %s, must be one-dimensionnal, of type float, and "\
		    "have matching dimension with A",name);
	goto err;
    }
    return PI;
 err:
    if (PI) {
	Py_DECREF(PI);
    }
    return NULL;
}

static PyArrayObject* MakeResult1( PyArrayObject* R, long M )
{
    int dimensions[1];
    if (R) {
	if (R->nd != 1 || R->descr->type_num != PyArray_DOUBLE
	    || R->dimensions[0]!=M ) {
	    PyErr_SetString(PyExc_ValueError,
			    "Array S, must be two-dimensionnal and of type float");
	    return NULL;
	}
	Py_INCREF(R);
    } else {
	dimensions[0]=M;
	R = (PyArrayObject*)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
	if (R==NULL) {
	    return NULL;
	}
    }
    return R;
}

static PyArrayObject* MakeResult2( PyArrayObject* R, long M, long N )
{
    int dimensions[2];
    if (R) {
	if (R->nd != 2 || R->descr->type_num != PyArray_DOUBLE
	    || R->dimensions[0]!=M || R->dimensions[1]!=N) {
	    PyErr_SetString(PyExc_ValueError,
			    "Array R, must be two-dimensionnal and of type float");
	    return NULL;
	}
	Py_INCREF(R);
    } else {
	dimensions[0]=M;
	dimensions[1]=N;
	R = (PyArrayObject*)PyArray_FromDims(2,dimensions,PyArray_DOUBLE);
	if (R==NULL) {
	    return NULL;
	}
    }
    return R;
}

static PyArrayObject* MakeResult3( PyArrayObject* R, long T, long M, long N )
{
    int dimensions[3];
    if (R) {
	if (R->nd != 3 || R->descr->type_num != PyArray_DOUBLE
	    || R->dimensions[0]!=T
	    || R->dimensions[1]!=M
	    || R->dimensions[2]!=N ) {
	    PyErr_SetString(PyExc_ValueError,
			    "Array Ksi, must be three-dimensionnal and of type float");
	    return NULL;
	}
	Py_INCREF(R);
    } else {
	dimensions[0]=T;
	dimensions[1]=M;
	dimensions[2]=N;
	R = (PyArrayObject*)PyArray_FromDims(3,dimensions,PyArray_DOUBLE);
	if (R==NULL) {
	    return NULL;
	}
    }
    return R;
}

static PyArrayObject* check_AL( PyObject* oA, long M, long N, char *name )
{ 
    PyArrayObject *A;
    A = (PyArrayObject*)PyArray_ContiguousFromObject(oA,PyArray_DOUBLE,2,2);
    if (A==NULL || A->dimensions[0]!=M || A->dimensions[1]!=N) {
	throwexcept("Array %s must be two-dimensionnal, of type float, and have matching dimension with A",name);
	goto err;
    }
    return A;
 err:
    if (A) {
	Py_DECREF(A);
    }
    return NULL;
}

/* --------------------------------------------------------------
   alpha_scaled
   Compute P(O1,...Ot,qt=Si|M) from a markov model
   M=(A,B,pi)
   input arguments:
     A the state transition probabilities
     B the probability matrix of each observation for each state
       (this is not the matrix B from the model but B[O1,...OT])
     pi The initial probability matrix
   --------------------------------------------------------------
*/
static void hmm_alpha_scaled( double *A, int A0, int A1,
			      double *B, int B0, int B1,
			      double *PI, int PI0,
			      double *R, int R0, int R1,
			      double *S, int S0,
			      int N, int T)
{
    int i,j,t;
    double *r,*b,*a,*pi,*r1,*s;
    double sum;
    r=R;
    a=A;
    b=B;
    pi=PI;
    s=S;
    *s=0;
    /* Compute alpha(1,i) */
    for(i=0;i<N;++i) {
	*r=(*b)*(*pi);
	*s+=*r;
	r+=R1;
	b+=B1;
	pi+=PI0;
    }
    *s=1./(*s);
    /* Rescale alpha(1,i) */
    for(i=0,r=R;i<N;++i,r+=R1)
	*r *= *s;
    for(t=1;t<T;++t) {
	b=B+t*B0;
	r=R+t*R0;
	s+=S0;
	for(j=0;j<N;++j) {
	    a=A+j*A1;
	    r1=R+(t-1)*R0;
	    sum=0;
	    for(i=0;i<N;++i) {
		sum+=(*a)*(*r1);
		a+=A0;
		r1+=R1;
	    }
	    *r=sum*(*b);
	    *s+=*r;
	    r+=R1;
	    b+=B1;
	}
	*s=1./(*s);
	/* Rescale alpha(1,i) */
	r=R+t*R0;
	for(i=0;i<N;++i,r+=R1)
	    *r *= *s;
    }
}

static PyObject* Py_hmm_alpha_scaled(PyObject *self, PyObject *args)
{
    PyObject *oA,*oB,*oPI,*result;
    PyArrayObject *A,*B,*PI,*R,*S;
    long N,T;

    R = NULL;
    S = NULL;
    if (!PyArg_ParseTuple(args,"OOO|O!O!",
			  &oA,&oB,&oPI,
			  &PyArray_Type,&R,
			  &PyArray_Type,&S))
	return NULL;
    
    if (!(A = check_square(oA,&N,"A"))) return NULL;
    if (!(B = check_second_dim(oB,&T,N,"B"))) return NULL;
    if (!(PI= check_one_dim(oPI,N,"PI"))) return NULL;
    if (!(R = MakeResult2(R,T,N))) return NULL;
    if (!(S = MakeResult1(S,T))) return NULL;
    
    hmm_alpha_scaled( (double*)A->data, STRIDE(A,0),STRIDE(A,1),
		      (double*)B->data, STRIDE(B,0),STRIDE(B,1),
		      (double*)PI->data, STRIDE(PI,0),
		      (double*)R->data, STRIDE(R,0),STRIDE(R,1),
		      (double*)S->data, STRIDE(S,0),
		      N,T);
    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(PI);
    result = Py_BuildValue("OO",R,S);
    Py_DECREF(R);
    Py_DECREF(S);
    return result;
}


/* --------------------------------------------------------------
   beta_scaled
   Compute P(qt=Si,O(t+1),..,O(T)|M) from a markov model using
   scaling factors
   M=(A,B,pi)
   input arguments:
     A the state transition probabilities
     B the probability matrix of each observation for each state
       (this is not the matrix B from the model but B[O1,...OT])
     pi The initial probability matrix
   --------------------------------------------------------------
*/
static void hmm_beta_scaled( double *A, int A0, int A1,
			     double *B, int B0, int B1,
			     double *S, int S0,
			     double *R, int R0, int R1,
			     int N, int T)
{
    int i,j,t;
    double *r,*b,*a,*tmp,*tmp0,*s;
    tmp0=R+(T-1)*R0;
    a=A;
    b=B;
    /* beta(T)(i)=1 */
    r=tmp0;
    s=S+(T-1)*S0;
    for(i=0;i<N;++i) {
	*r=*s;
	r+=R1;
    }
    
    for(t=T-2;t>=0;--t) {
	b=B+(t+1)*B0;
	/* we use beta(T) as a temp to hold beta(t+1)*B(t+1) */
	tmp=tmp0;
	r=R+(t+1)*R0;
	s-=S0;
	for(j=0;j<N;++j) {
	    *tmp = (*r)*(*b)*(*s);
	    b+=B1;
	    r+=R1;
	    tmp+=R1;
	}
	/* Now we compute beta(t,i)=sum(Aij*tmp(j)) */
	r=R+t*R0;
	for(i=0;i<N;++i) {
	    a=A+i*A0;
	    *r=0;
	    tmp=tmp0;
	    for(j=0;j<N;++j) {
		(*r)+=(*a)*(*tmp);
		a+=A1;
		tmp+=R1;
	    }
	    r+=R1;
	}
    }
    r=tmp0;
    s=S+(T-1)*S0;
    for(i=0;i<N;++i) {
	*r=*s;
	r+=R1;
    }
}

static PyObject* Py_hmm_beta_scaled(PyObject *self, PyObject *args)
{
    PyObject *oA,*oB,*oS;
    PyArrayObject *A,*B,*S,*R;
    long N,T;

    R = NULL;
    if (!PyArg_ParseTuple(args,"OOO|O!",
			  &oA,&oB,&oS,
			  &PyArray_Type,&R ))
	return NULL;
    
    if (!(A = check_square(oA,&N,"A"))) return NULL;
    if (!(B = check_second_dim(oB,&T,N,"B"))) return NULL;
    if (!(S = check_one_dim(oS,T,"S"))) return NULL;
    if (!(R = MakeResult2(R,T,N))) return NULL;

    hmm_beta_scaled( (double*)A->data, STRIDE(A,0),STRIDE(A,1),
	      (double*)B->data, STRIDE(B,0),STRIDE(B,1),
	      (double*)S->data, STRIDE(S,0),
	      (double*)R->data, STRIDE(R,0),STRIDE(R,1),
	      N,T);
    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(S);
    return PyArray_Return(R);
}

/* --------------------------------------------------------------
   ksi
   Compute ksi(t,i,j)
   --------------------------------------------------------------
*/
static void hmm_ksi( double *A, int A0, int A1,
		     double *B, int B0, int B1,
		     double *AL, int AL0, int AL1,
		     double *BE, int BE0, int BE1,
		     double *R, int R0, int R1, int R2,
		     int N, int T)
{
    int t,i,j;
    double sum;
    double *r,*a,*b,*al,*be;
    for(t=0;t<T-1;++t) {
	al=AL+t*AL0;
	sum=0;
	for(i=0;i<N;++i) {
	    r=R+t*R0+i*R1;
	    a=A+i*A0;
	    b=B+(t+1)*B0;
	    be=BE+(t+1)*BE0;
	    for(j=0;j<N;++j) {
		*r = (*al)*(*a)*(*b)*(*be);
		sum+=*r;
		r+=R2;
		a+=A1;
		b+=B1;
		be+=BE1;
	    }
	    al+=AL1;
	}
	if (sum) {
	    sum=1.0/sum;
	    for(i=0;i<N;++i) {
		r=R+t*R0+i*R1;
		for(j=0;j<N;++j) {
		    (*r)*=sum;
		    r+=R2;
		}
	    }
	}
    }
}

static PyObject* Py_hmm_ksi(PyObject *self, PyObject *args)
{
    PyObject *oA,*oB,*oAL,*oBE;
    PyArrayObject *A,*B,*AL,*BE,*R;
    PyObject *ret=NULL;
    long N,T;

    R = NULL;
    if (!PyArg_ParseTuple(args,"OOOO|O!",
			  &oA,&oB,&oAL,&oBE,
			  &PyArray_Type,&R ))
	return NULL;
    
    if (!(A = check_square(oA,&N,"A"))) return NULL;
    if (!(B = check_second_dim(oB,&T,N,"B"))) goto exit_a;
    if (!(BE= check_AL(oBE,T,N,"BE"))) goto exit_b;
    if (!(AL= check_AL(oAL,T,N,"AL"))) goto exit_be;
    if (!(R = MakeResult3(R,T-1,N,N))) goto exit_al;

    hmm_ksi( (double*)A->data, STRIDE(A,0),STRIDE(A,1),
	     (double*)B->data, STRIDE(B,0),STRIDE(B,1),
	     (double*)AL->data, STRIDE(AL,0), STRIDE(AL,1),
	     (double*)BE->data, STRIDE(BE,0), STRIDE(BE,1),
	     (double*)R->data, STRIDE(R,0),STRIDE(R,1),STRIDE(R,2),
	     N,T);
    ret = PyArray_Return(R);
 exit_al:
    Py_DECREF(AL);
 exit_be:
    Py_DECREF(BE);
 exit_b:
    Py_DECREF(B);
 exit_a:
    Py_DECREF(A);
    return ret;
}

/* --------------------------------------------------------------
   _update_iter_B
   Updates the observed probabilities in B
   --------------------------------------------------------------
*/
static void hmm_update_iter_B( double *G, int G0, int G1,
			       long *Obs, int Obs0,
			       double *B, int B0, int B1,
			       int N, int T, int M)
{
    int i,j,t;
    double *b,*g;
    long *o;

    o = Obs;
    for(t=0;t<T;++t) {
	i=*o;
	o+=Obs0;
	b=B+i*B0;
	g=G+t*G0;
	for(j=0;j<N;++j) {
	    *b+=*g;
	    b+=B1;
	    g+=G1;
	}
    }
}

static PyObject* Py_hmm_update_iter_B(PyObject *self, PyObject *args)
{
    PyObject *oG,*oB,*oObs;
    PyArrayObject *G,*B,*Obs;
    long N,T,M;

    if (!PyArg_ParseTuple(args,"OOO",&oG,&oObs,&oB))
	return NULL;
    
    /*    printf("g=%p\nobs=%p\nB=%p\n",oG,oObs,oB);*/
    if (!(G = check_2D(oG,&T,&N,"G"))) return NULL;
    if (!(Obs = check_long_one_dim(oObs,T,"Obs"))) return NULL;
    if (!(B = check_second_dim(oB,&M,N,"B"))) return NULL;
    /*    printf("g'=%p\nobs'=%p\nB'=%p\n",G,Obs,B);*/
      
    hmm_update_iter_B( (double*)G->data, STRIDE(G,0),STRIDE(G,1),
		       (long*)Obs->data, STRIDE_LONG(Obs,0),
		       (double*)B->data, STRIDE(B,0),STRIDE(B,1),
		       N,T,M);
    Py_DECREF(G);
    Py_DECREF(Obs);
    Py_DECREF(B);
    Py_INCREF(Py_None);
    return Py_None;
}

/* --------------------------------------------------------------
   _hmm_correctm
   helper function to replace lines or columns of zeros with a
   single value
   --------------------------------------------------------------
*/
static void hmm_correctm( double *G, int G0, int G1, double v, int M, int N )
{
    int i,j;
    double *g;
    double sum;

    for(i=0;i<M;++i) {
	g = G+i*G0;
	sum=0;
	for(j=0;j<N;++j) {
	    sum+=*g;
	    g+=G1;
	}
	if (sum==0.0) {
	    g = G+i*G0;
	    for(j=0;j<N;++j) {
		*g=v;
		g+=G1;
	    }
	}
    }
}
static PyObject* Py_hmm_correctm(PyObject *self, PyObject *args)
{
    PyArrayObject *R;
    long idx;
    double v=0.0;
    int M,N;

    R = NULL;
    if (!PyArg_ParseTuple(args,"O!ld",&PyArray_Type, &R, &idx, &v ))
	return NULL;
    R = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)R,PyArray_DOUBLE,2,2);

    if (!R) {
	    PyErr_SetString(PyExc_ValueError,"Array must be of type float, and 2 dimensionnal");
	    Py_DECREF(R);
	    return NULL;
    }
    M = R->dimensions[0];
    N = R->dimensions[1];
    if (idx==1) {
	hmm_correctm((double*)R->data,STRIDE(R,0),STRIDE(R,1),v,M,N);
    } else if (idx==0) {
	hmm_correctm((double*)R->data,STRIDE(R,1),STRIDE(R,0),v,N,M);
    } else {
	PyErr_SetString(PyExc_ValueError,"Index out of range");
	Py_DECREF(R);
	return NULL;
    }
    return PyArray_Return(R);
}

/* --------------------------------------------------------------
   _hmm_normalize_B
   helper function to replace lines or columns of zeros with a
   single value
   --------------------------------------------------------------
*/
static void hmm_normalize_B( double *B, int B0, int B1,
			     double *V, long M, long N )
{
    long i,j;
    double *b,*v,*W,*w;

    W = (double*)malloc(N*sizeof(double));
    if (!W) return;
    w = W;
    v = V;
    for(j=0;j<N;++j) {
	if (*v)
	    *w=1./(*v);
	else
	    *w=1.0;
	w++;
	v++;
    }
    for(i=0;i<M;++i) {
	b = B+i*B0;
	w = W;
	for(j=0;j<N;++j) {
	    *b *= *w;
	    b+=B1;
	    ++w;
	}
    }
    free(W);
}
static PyObject* Py_hmm_normalize_B(PyObject *self, PyObject *args)
{
    PyObject *oB,*oV;
    PyArrayObject *B,*V;
    long M,N;

    if (!PyArg_ParseTuple(args,"OO",&oB,&oV ))
	return NULL;
    if (!(B = check_2D(oB,&M,&N,"B"))) return NULL;
    if (!(V = check_one_dim(oV,N,"V"))) {
	Py_DECREF(B);
	return NULL;
    }
    hmm_normalize_B((double*)B->data,STRIDE(B,0),STRIDE(B,1),
		    (double*)V->data,M,N);
    Py_INCREF(Py_None);
    return Py_None;
}


/* --------------------------------------------------------------
   _array_set
   helper function to do a memset on an array
   --------------------------------------------------------------
*/
static PyObject* Py_array_set(PyObject *self, PyObject *args)
{
    PyArrayObject *R;
    long N,i;
    double v=0.0,*p,*pe;

    R = NULL;
    if (!PyArg_ParseTuple(args,"O!|d",&PyArray_Type, &R, &v ))
	return NULL;
    R = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)R,PyArray_DOUBLE,R->nd,R->nd);

    if (!R) {
	    PyErr_SetString(PyExc_ValueError,"Array must be of type float");
	    return NULL;
    }
    for(i=0,N=1;i<R->nd;++i) {
	N*=R->dimensions[i];
    }
    p=(double*)R->data;
    pe=p+N;
    for(;p<pe;++p) *p=v;
    return PyArray_Return(R);
}

/* --------------------------------------------------------------
   _array_allclose
   Returns One if array A and B are close according to ATOL and RTOL
   A and B must be of type PyArray_DOUBLE
   --------------------------------------------------------------
*/
#define dist(x,y) ( x<y ? y-x : x-y)
#define abs(x) (x<0 ? -x : x )
static int array_allclose( double *X,double *Y, int N, double atol, double rtol )
{
    double *x,*xe,*y;
    double d, rel;
    xe = X+N;
    for(x=X,y=Y;x<xe;++x,++y) {
	d = dist(*x,*y);
	rel = atol+abs(*y)*rtol;
	if (d>rel) return 0;
    }
    return 1;
}
#undef dist
#undef abs

static PyObject* Py_array_allclose(PyObject *self, PyObject *args)
{
    PyArrayObject *A,*B;
    PyObject *oA,*oB;
    long N,i;
    double atol,rtol;
    int r;

    atol=1e-8;
    rtol=1e-8;
    if (!PyArg_ParseTuple(args,"OO|dd",&oA,&oB, &rtol,&atol ))
	return NULL;
    A = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)oA,PyArray_DOUBLE,1,0);
    if (!A) {
	    PyErr_SetString(PyExc_ValueError,"Array must be of type float");
	    return NULL;
    }
    B = (PyArrayObject*)PyArray_ContiguousFromObject((PyObject*)oB,PyArray_DOUBLE,1,0);
    if (!B) {
	    PyErr_SetString(PyExc_ValueError,"Array must be of type float");
	    goto error;
    }
    if (A->nd!=B->nd) {
	    PyErr_SetString(PyExc_ValueError,"Arrays must have the same number of dimensions");
	    goto error;
    }
    N=1;
    for(i=0;i<A->nd;++i) {
	if (A->dimensions[i]!=B->dimensions[i]) {
	    PyErr_SetString(PyExc_ValueError,"Arrays must have the same dimensions");
	    goto error;
	}
	N*=A->dimensions[i];
    }
    r=array_allclose((double*)A->data,(double*)B->data,N,atol,rtol);
    Py_DECREF(A);
    Py_DECREF(B);
    return Py_BuildValue("i",r);
 error:
    Py_XDECREF(A);	
    Py_XDECREF(B);
    return NULL;
}

/* -----------------------------------------------------------------
   MODULE DECLARATION
   ----------------------------------------------------------------- */
static PyMethodDef HmmMethods[] = {
    {"_hmm_alpha_scaled",  Py_hmm_alpha_scaled, METH_VARARGS},
    {"_hmm_beta_scaled", Py_hmm_beta_scaled, METH_VARARGS},
    {"_hmm_ksi", Py_hmm_ksi, METH_VARARGS},
    {"_hmm_update_iter_B", Py_hmm_update_iter_B, METH_VARARGS},
    {"_hmm_correctm", Py_hmm_correctm, METH_VARARGS},
    {"_hmm_normalize_B", Py_hmm_normalize_B, METH_VARARGS},
    {"_array_set", Py_array_set, METH_VARARGS},
    {"_array_allclose", Py_array_allclose, METH_VARARGS},
    {NULL,      NULL}        /* Sentinel */
};

DL_EXPORT(void)
init_hmm_c()
{
    PyObject* hmm_module;
    
    hmm_module = Py_InitModule3("_hmm_c", HmmMethods,
				"This module contains helper functions for "\
				"the hmm python module");
    PyModule_AddStringConstant(hmm_module,"__revision__",__revision__);   
    import_array();
}
