error2name = {
    # Converged but degraded solution (e.g. WDLS with psuedo-inverse singular)
    1:"E_DEGRADED",

    # No error
    0:"E_NOERROR",

    # Failed to converge
    -1:"E_NO_CONVERGE",

    # Undefined value (e.g. computed a NAN, or tan(90 degrees) )
    -2:"E_UNDEFINED",

    # Chain size changed
    -3:"E_NOT_UP_TO_DATE",

    # Input size does not match internal state
    -4:"E_SIZE_MISMATCH",

    # Maximum number of iterations exceeded
    -5:"E_MAX_ITERATIONS_EXCEEDED",

    # Requested index out of range
    -6:"E_OUT_OF_RANGE",

    # Not yet implemented
    -7:"E_NOT_IMPLEMENTED",

    # Internal svd calculation failed
    -8:"E_SVD_FAILED"
}
