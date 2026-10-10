"""Exact coefficient-encoding error for the fused hi+lo representation."""
from fractions import Fraction as F
import numpy as np
from c2fuse import norm_upper
from sensitivity_c2fuse import derivative_tail_bound

def encoding_certificate(compiled,*,paired=True):
    plan=compiled['plan'];norms=[]
    for coefficient in plan['coefficients']:
        error=[]
        for row in coefficient:
            values=[]
            for value in row:
                high=float(value);low=float(F(value)-F(high)) if paired else 0.
                values.append(F(value)-F(high)-F(low))
            error.append(values)
        norms.append(norm_upper(np.array(error,dtype=object)))
    rho=plan.get('parameter_norm_radius',plan['norm_radius'])
    radius=plan.get('parameter_numerical_radius',plan['radius'])
    bound=min(sum(v*rho**k for k,v in enumerate(norms)),plan['constant']*sum(v*radius**k for k,v in enumerate(norms)))
    penalty=compiled['condition_penalty']
    derivative=derivative_tail_bound(norms,-1,rho,radius,plan['direction_norm_bound'],plan['constant'])[0] if 'direction' in plan else F(0)
    return dict(coefficient_error_bound=penalty*bound,
        value_tail_and_encoding_bound=compiled['original_error_bound']+penalty*bound,
        derivative_coefficient_error_bound=penalty*derivative,
        derivative_tail_and_encoding_bound=compiled['original_derivative_error_bound']+penalty*derivative if 'direction' in plan else None,
        scope='Exact arithmetic on encoded hi+lo coefficients; execution rounding and underflow are additional')
