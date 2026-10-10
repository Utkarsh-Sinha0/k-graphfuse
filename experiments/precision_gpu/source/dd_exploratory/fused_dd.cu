struct DD { double hi,lo; };
__device__ DD add(DD a, DD b) {
    double s=a.hi+b.hi, v=s-a.hi;
    double t=((b.hi-v)+(a.hi-(s-v)))+a.lo+b.lo;
    double h=s+t;
    return {h,t-(h-s)};
}
__device__ DD mul(double a, DD b) {
    double h=a*b.hi;
    double l=fma(a,b.hi,-h)+a*b.lo;
    double s=h+l;
    return {s,l-(s-h)};
}
extern "C" __global__ void scale_input(const double* x,const double* scale,double* incoming,int count,int q) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t<count) incoming[t]=x[t]/scale[t/q];
}
extern "C" __global__ void horner(
    const int* rows,const int* columns,const double* edges,const double* x,
    const double* ch,const double* cl,const double* eh,const double* el,
    const double* dh,const double* dl,double* oh,double* ol,double* odh,double* odl,
    const double* direction,int count,int q,int k,int initial,int sensitivity,double theta) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=count) return;
    int row=t/q, channel=t%q;
    DD y={0.,0.},dy={0.,0.};
    if(!initial) {
        for(int p=rows[row];p<rows[row+1];++p) {
            int index=columns[p]*q+channel;
            y=add(y,mul(edges[p],{eh[index],el[index]}));
            if(sensitivity) dy=add(dy,mul(edges[p],{dh[index],dl[index]}));
        }
        if(sensitivity) {
            DD directional=mul(direction[row],{eh[t],el[t]});
            y=add(y,mul(theta,directional));
            dy=add(add(dy,directional),mul(theta,mul(direction[row],{dh[t],dl[t]})));
        }
    }
    for(int j=0;j<q;++j) {
        int index=(k*q+channel)*q+j;
        y=add(y,mul(x[row*q+j],{ch[index],cl[index]}));
    }
    oh[t]=y.hi;ol[t]=y.lo;
    if(sensitivity) { odh[t]=dy.hi;odl[t]=dy.lo; }
}
extern "C" __global__ void finish(const double* hi,const double* lo,const double* scale,double* output,int count,int q) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t<count) output[t]=(hi[t]+lo[t])*scale[t/q];
}
