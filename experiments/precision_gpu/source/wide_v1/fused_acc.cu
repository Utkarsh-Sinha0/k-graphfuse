struct Acc { double sum,error; };
template<int COMP> __device__ void product(Acc& a,double x,double y,double correction=0.) {
if(COMP) {
    double p=x*y, t=a.sum+p;
    double e=fabs(a.sum)>=fabs(p) ? (a.sum-t)+p : (p-t)+a.sum;
    a.error+=e+fma(x,y,-p)+correction;
    a.sum=t;
} else {
    a.sum=fma(x,y,a.sum);
    a.error+=correction;
}
}
__device__ void value_product(Acc& a,double x,double y,double correction,int k) {
    if(FINAL_COMPENSATE && k==0) product<1>(a,x,y,correction);
    else product<COMPENSATE>(a,x,y,correction);
}
extern "C" __global__ void horner(
    const int* rows,const int* columns,const double* edges,const double* x,
    const double* ch,const double* cl,const double* eh,const double* el,
    const double* dh,const double* dl,double* oh,double* ol,double* odh,double* odl,
    const double* direction,int count,int q,int k,int initial,int sensitivity,double theta) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=count) return;
    int row=t/q, channel=t%q;
    Acc y={0.,0.},dy={0.,0.};
    if(!initial) {
        for(int p=rows[row];p<rows[row+1];++p) {
            int index=columns[p]*q+channel;
            value_product(y,edges[p],eh[index],0.,k);
            if(sensitivity) product<DERIVATIVE_COMPENSATE>(dy,edges[p],dh[index]);
        }
        if(sensitivity) {
            product<DERIVATIVE_COMPENSATE>(dy,direction[row],eh[t]);
            if(theta!=0.) {
                value_product(y,theta*direction[row],eh[t],0.,k);
                product<DERIVATIVE_COMPENSATE>(dy,theta*direction[row],dh[t]);
            }
        }
    }
#if PROJECTED
    value_product(y,1.,x[(row*STAGES+k)*q+channel],0.,k);
#else
    for(int j=0;j<q;++j) {
        int index=(k*q+channel)*q+j;
        value_product(y,x[row*q+j],ch[index],x[row*q+j]*cl[index],k);
    }
#endif
    oh[t]=y.sum+y.error;
    if(sensitivity) odh[t]=dy.sum+dy.error;
}
extern "C" __global__ void finish(const double* hi,const double* lo,const double* scale,double* output,int count,int q) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t<count) output[t]=hi[t]*scale[t/q];
}
