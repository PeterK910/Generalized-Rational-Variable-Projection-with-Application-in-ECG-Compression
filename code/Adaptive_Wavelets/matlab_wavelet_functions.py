import numpy as np
import pywt

# NOTE
# The functions below have been translated from MATLAB's own factory functions,
# and have not been under rigorous testing!

def wavedec(x, n, IN3, IN4=None):
    """
    WAVEDEC performs a multilevel 1-D wavelet analysis
    using either a specific wavelet 'wname' or a specific set 
    of wavelet decomposition filters (see WFILTERS).

    [C,L] = WAVEDEC(X,N,'wname') returns the wavelet decomposition of the 
    signal X at level N, using 'wname'. WAVEDEC does not enforce a maximum
    level restriction. Use WMAXLEV to ensure the wavelet coefficients are 
    free from boundary effects. If boundary effects are not a concern in 
    your application, a good rule is to set N less than or equal to 
    fix(log2(length(X))).

    The output vector, C, contains the wavelet decomposition. L contains
    the number of coefficients by level.
    C and L are organized as:
    C      = [app. coef.(N)|det. coef.(N)|... |det. coef.(1)]
    L(1)   = length of app. coef.(N)
    L(i)   = length of det. coef.(N-i+2) for i = 2,...,N+1
    L(N+2) = length(X).

    [C,L] = WAVEDEC(X,N,Lo_D,Hi_D) Lo_D is the decomposition low-pass 
    filter and Hi_D is the decomposition high-pass filter.


    See also DWT, WAVEINFO, WAVEREC, WFILTERS, WMAXLEV.

    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 30-Jul-2008.
    Copyright 1995-2008 The MathWorks, Inc.
    """

    # Check arguments.
    if isinstance(IN3, str):
        # IN3 is a wavelet name.
        wavelet = pywt.Wavelet(IN3)
        Lo_D = wavelet.dec_lo
        Hi_D = wavelet.dec_hi
    else:
        # IN3 and IN4 are filter coefficients.
        Lo_D = IN3
        Hi_D = IN4
        if Hi_D is None:
            raise ValueError("High-pass decomposition filter IN4 must be provided when IN3 is not a wavelet name.")

    # Initialization.
    """
    s=x.shape
    print(s)
    x=x.T[0]
    print(x)
    """
    
    c = []
    l = np.zeros(n + 2, dtype=int)
    if len(x)==0:
        return c, l

    l[-1] = len(x)

    # Decomposition.
    for k in range(1, n + 1):
        x, d = dwt(x, Lo_D, Hi_D)
        c = np.concatenate([d, c])  # store detail coefficients
        l[n + 1 - k] = len(d)  # store length of detail coefficients
    
    # Last approximation.
    c = np.concatenate([x, c])
    l[0] = len(x)
    """
    if s[0]>1:
        c=np.reshape(c, (len(c), 1))
        l=np.reshape(l, (len(l), 1))
        print(l)
    """

    return c, l

def dwt(x, *args, mode='symmetric', shift=0):
    """
    DWT performs a single-level 1-D wavelet decomposition
    with respect to either a particular wavelet ('wname',
    see WFILTERS for more information) or particular wavelet filters
    (Lo_D and Hi_D) that you specify.

    [CA,CD] = DWT(X,'wname') computes the approximation
    coefficients vector CA and detail coefficients vector CD,
    obtained by a wavelet decomposition of the vector X.
    'wname' is a character vector containing the wavelet name.

    [CA,CD] = DWT(X,Lo_D,Hi_D) computes the wavelet decomposition
    as above given these filters as input:
    Lo_D is the decomposition low-pass filter.
    Hi_D is the decomposition high-pass filter.
    Lo_D and Hi_D must be the same length.

    Let LX = length(X) and LF = the length of filters; then
    length(CA) = length(CD) = LA where LA = CEIL(LX/2),
    if the DWT extension mode is set to periodization.
    LA = FLOOR((LX+LF-1)/2) for the other extension modes.  
    For the different signal extension modes, see DWTMODE. 

    [CA,CD] = DWT(...,'mode',MODE) computes the wavelet 
    decomposition with the extension mode MODE you specify.
    MODE is a character vector containing the extension mode.

    Example:
        x = 1:8;
        [ca,cd] = dwt(x,'db1','mode','sym')

    See also DWTMODE, IDWT, WAVEDEC, WAVEINFO.

    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 06-Feb-2011.
    Copyright 1995-2015 The MathWorks, Inc.
    """

    if isinstance(args[0], str):
        # args[0] is a wavelet name
        wavelet = pywt.Wavelet(args[0])
        Lo_D = wavelet.dec_lo
        Hi_D = wavelet.dec_hi
    else:
        if len(args) < 2:
            raise ValueError("Both Lo_D and Hi_D filters must be provided.")
        Lo_D = args[0]
        Hi_D = args[1]

        # Validate Lo_D and Hi_D
        if len(Lo_D) < 2 or len(Hi_D) < 2:
            raise ValueError("Invalid filter length: Filters must have at least 2 coefficients.")

    # Compute sizes and shape
    lf = len(Lo_D)
    lx = len(x)

    # Handle extension modes
    if mode == 'per':
        lenEXT = lf // 2
        last = 2 * int(np.ceil(lx / 2))
    else:
        lenEXT = lf - 1
        last = lx + lf - 1

    # Extend the signal
    y = pywt.pad(x, pad_widths=(lenEXT,), mode=mode)

    # Compute coefficients of approximation
    z = np.convolve(y, Lo_D, mode='valid')
    first = 2 - shift
    a = z[first - 1:last:2]

    # Compute coefficients of detail
    z = np.convolve(y, Hi_D, mode='valid')
    d = z[first - 1:last:2]

    return a, d

def waverec(c, l, *args):
    """
    WAVEREC performs a multilevel 1-D wavelet reconstruction
    using either a specific wavelet ('wname', see WFILTERS) or
    specific reconstruction filters (Lo_R and Hi_R).

    X = WAVEREC(C,L,'wname') reconstructs the signal X
    based on the multilevel wavelet decomposition structure
    [C,L] (see WAVEDEC).

    For X = WAVEREC(C,L,Lo_R,Hi_R),
    Lo_R is the reconstruction low-pass filter and
    Hi_R is the reconstruction high-pass filter.

    See also APPCOEF, IDWT, WAVEDEC.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 06-Feb-2011.
    Copyright 1995-2016 The MathWorks, Inc.
    """

    # Utilize appcoef with level 0 to reconstruct the full signal.
    return appcoef(c, l, *args, level=0)

def appcoef(c, l, *args, level=None):
    """
    APPCOEF computes the approximation coefficients of a
    one-dimensional signal.

    A = APPCOEF(C,L,'wname',N) computes the approximation
    coefficients at level N using the wavelet decomposition
    structure [C,L] (see WAVEDEC).
    'wname' is a character vector containing the wavelet name.
    Level N must be an integer such that 0 <= N <= length(L)-2. 

    A = APPCOEF(C,L,'wname') extracts the approximation
    coefficients at the last level length(L)-2.

    Instead of giving the wavelet name, you can give the filters.
    For A = APPCOEF(C,L,Lo_R,Hi_R) or
    A = APPCOEF(C,L,Lo_R,Hi_R,N),
    Lo_R is the reconstruction low-pass filter and
    Hi_R is the reconstruction high-pass filter.
    
    See also DETCOEF, WAVEDEC.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 06-Feb-2011.
    Copyright 1995-2015 The MathWorks, Inc.
    """

    # Validate inputs
    if isinstance(c, list):
        c = np.array(c)
    if isinstance(l, list):
        l = np.array(l)
        
    if isinstance(args[0], str):
        # args[0] is a wavelet name
        wavelet = pywt.Wavelet(args[0])
        Lo_R = wavelet.rec_lo
        Hi_R = wavelet.rec_hi
    else:
        if len(args) < 2:
            raise ValueError("Both Lo_R and Hi_R filters must be provided.")
        Lo_R = args[0]
        Hi_R = args[1]

    if level is None:
        level = len(l) - 2

    if level < 0 or level > len(l) - 2:
        raise ValueError("Invalid level: level must be between 0 and len(l)-2.")

    # Initialize the approximation coefficients at the highest level
    #a = c[:l[0, 0]]
    a = c[:l[0]]

    # Iteratively reconstruct the approximation coefficients down to the desired level
    imax = len(l) + 1
    for p in range(len(l) - 2, level, -1):
        d = detcoef(c, l, p)  # Extract detail coefficients
        #print('d', d)
        a = idwt(a, d, Lo_R, Hi_R, l[imax - p - 1])

    return a

def detcoef(coefs, longs, levels=None, as_cells=False):
    """
    D = DETCOEF(C,L,N) extracts the detail coefficients
    at level N from the wavelet decomposition structure [C,L].
    See WAVEDEC for more information on C and L.
    Level N must be an integer such that 1 <= N <= NMAX
    where NMAX = length(L)-2.

    D = DETCOEF(C,L) extracts the detail coefficients
    at last level NMAX.

    If N is a vector of integers such that 1 <= N(j) <= NMAX:

        DCELL = DETCOEF(C,L,N,'cells') returns a cell array where
        DCELL{j} contains the coefficients of detail N(j).

        If length(N)>1, DCELL = DETCOEF(C,L,N) is equivalent to
        DCELL = DETCOEF(C,L,N,'cells').

        DCELL = DETCOEF(C,L,'cells') is equivalent to 
        DCELL = DETCOEF(C,L,[1:NMAX])

        [D1,...,Dp] = DETCOEF(C,L,[N(1),...,N(p)]) extracts the details
        coefficients at levels [N(1),...,N(p)].

    See also APPCOEF, WAVEDEC.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 20-Dec-2010.
    Copyright 1995-2010 The MathWorks, Inc.
    """

    # Validate inputs
    coefs = np.asarray(coefs)
    longs = np.asarray(longs)
    
    if levels is None:
        levels = len(longs) - 2
    
    nmax = len(longs) - 2

    if isinstance(levels, int):
        levels = [levels]

    if isinstance(levels, list):
        if any(level < 1 or level > nmax for level in levels):
            raise ValueError("Invalid level values: Levels must be between 1 and nmax (inclusive).")
    else:
        raise ValueError("Levels must be an integer or a list of integers.")

    # Compute the first and last indices of each detail coefficient
    first = np.cumsum(longs) + 1
    first = first[-3::-1]  # Reverse order except last one
    longs = longs[-2:0:-1]  # Reverse and slice to remove first and last
    #longs=longs.T[0]
    last = first + longs-1
#    print(first)
#    print(longs)
#    print(last)

    # Extract the detail coefficients at specified levels
    details = [coefs[first[level-1]-1:last[level-1]] for level in levels]

    if len(details) == 1 and not as_cells:
        return details[0]
    else:
        return details

def idwt(a, d, *args, mode='sym', shift=0):
    """
    IDWT performs a single-level 1-D wavelet reconstruction
    with respect to either a particular wavelet
    ('wname', see WFILTERS for more information) or particular wavelet
    reconstruction filters (Lo_R and Hi_R) that you specify.

    X = IDWT(CA,CD,'wname') returns the single-level
    reconstructed approximation coefficients vector X
    based on approximation and detail coefficients
    vectors CA and CD, and using the wavelet 'wname'.

    X = IDWT(CA,CD,Lo_R,Hi_R) reconstructs as above,
    using filters that you specify:
    Lo_R is the reconstruction low-pass filter.
    Hi_R is the reconstruction high-pass filter.
    Lo_R and Hi_R must be the same length.

    Let LA = length(CA) = length(CD) and LF the length
    of the filters; then length(X) = LX where LX = 2*LA
    if the DWT extension mode is set to periodization.
    LX = 2*LA-LF+2 for the other extension modes.
    For the different DWT extension modes, see DWTMODE. 

    X = IDWT(CA,CD,'wname',L) or X = IDWT(CA,CD,Lo_R,Hi_R,L)
    returns the length-L central portion of the result
    obtained using IDWT(CA,CD,'wname'). L must be less than LX.

    X = IDWT(...,'mode',MODE) computes the wavelet
    reconstruction using the specified extension mode MODE.

    X = IDWT(CA,[], ... ) returns the single-level
    reconstructed approximation coefficients vector X
    based on approximation coefficients vector CA.
    
    X = IDWT([],CD, ... ) returns the single-level
    reconstructed detail coefficients vector X
    based on detail coefficients vector CD.
    
    See also DWT, DWTMODE, UPWLEV.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 06-Feb-2011.
    Copyright 1995-2015 The MathWorks, Inc.
    """

    # Check if either a or d is empty
    if a is None or len(a) == 0:
        if d is None or len(d) == 0:
            raise ValueError("At least one of 'a' or 'd' must be non-empty.")
        validate_input(d, 'CD')
    elif d is None or len(d) == 0:
        validate_input(a, 'CA')
    else:
        validate_input(a, 'CA')
        validate_input(d, 'CD')

    # Determine the filters
    if isinstance(args[0], str):
        wavelet = pywt.Wavelet(args[0])
        Lo_R = wavelet.rec_lo
        Hi_R = wavelet.rec_hi
        next_arg = 1
    else:
        if len(args) < 2:
            raise ValueError("Insufficient filter arguments provided.")
        Lo_R, Hi_R = np.array(args[0]), np.array(args[1])
        next_arg = 2

        # Validate filter lengths
        if len(Lo_R) < 2 or len(Hi_R) < 2 or len(Lo_R) % 2 != 0 or len(Hi_R) % 2 != 0:
            raise ValueError("Invalid filter lengths. Filters must have even length.")

    # Process additional arguments
    lx = None
    for i in range(next_arg, len(args)):
        if isinstance(args[i], str):
            if args[i] == 'mode':
                mode = args[i + 1]
            elif args[i] == 'shift':
                shift = args[i + 1] % 2
        else:
            lx = args[i]

    # Perform the inverse DWT
    x_approx = upsconv1(a, Lo_R, lx, mode, shift)
    x_detail = upsconv1(d, Hi_R, lx, mode, shift)

    return x_approx + x_detail

def validate_input(coeff, name):
    if not isinstance(coeff, (np.ndarray, list)) or np.any(np.isnan(coeff)) or np.any(np.isinf(coeff)):
        raise ValueError(f"Invalid {name} coefficients. Must be a finite, real-valued vector.")

def upsconv1(x, f, s=None, dwtARG1=None, dwtARG2=None):
    """
    UPSCONV1 Upsample and convolution 1D.

    Y = UPSCONV1(X,F_R,L,DWTATTR) returns the length-L central 
    portion of the one step dyadic interpolation (upsample and
    convolution) of vector X using filter F_R. The upsample 
    and convolution attributes are described by DWTATTR.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 06-May-2003.
    Last Revision: 21-May-2003.
    Copyright 1995-2004 The MathWorks, Inc.
    """

    # Special case: if input is empty, return 0
    if x is None or len(x) == 0:
        return np.zeros(s if s is not None else 1, dtype=float)

    # Check arguments for Extension and Shift
    if dwtARG1 is None and dwtARG2 is None:
        perFLAG = False
        dwtSHIFT = 0
    elif isinstance(dwtARG1, dict):
        perFLAG = (dwtARG1.get('extMode', 'sym') == 'per')
        dwtSHIFT = dwtARG1.get('shift1D', 0) % 2
    elif isinstance(dwtARG1, str):
        perFLAG = (dwtARG1 == 'per')
        dwtSHIFT = dwtARG2 % 2 if dwtARG2 is not None else 0

    # Define length
    lx = 2 * len(x)
    lf = len(f)
    if s is None:
        s = lx - lf + 2 if not perFLAG else lx

    # Compute Upsampling and Convolution
    y = np.array(x)
    if not perFLAG:
        #print(y)
        y = np.convolve(dyadup(y, 0), f, mode='full')
        y = wkeep1(y, s, 'c', dwtSHIFT)
    else:
        y = dyadup(y, 0, 1)
        y = np.pad(y, (lf // 2, lf // 2), mode='wrap')
        y = np.convolve(y, f, mode='full')
        y = y[lf:lf+s]
        if dwtSHIFT == 1:
            y = np.roll(y, -1)
    
    return y

def dyadup(x, evenodd=1, dim='c', evenLEN=False):
    """
    DYADUP Dyadic upsampling.
    DYADUP implements a simple zero-padding scheme very
    useful in the wavelet reconstruction algorithm.

    Y = DYADUP(X,EVENODD), where X is a vector, returns
    an extended copy of vector X obtained by inserting zeros.
    Whether the zeros are inserted as even- or odd-indexed
    elements of Y depends on the value of positive integer
    EVENODD:
    If EVENODD is even, then Y(2k-1) = X(k), Y(2k) = 0.
    If EVENODD is odd,  then Y(2k-1) = 0   , Y(2k) = X(k).

    Y = DYADUP(X) is equivalent to Y = DYADUP(X,1)

    Y = DYADUP(X,EVENODD,'type') or
    Y = DYADUP(X,'type',EVENODD) where X is a matrix,
    return extended copies of X obtained by inserting columns 
    of zeros (or rows or both) if 'type' = 'c' (or 'r' or 'm'
    respectively), according to the parameter EVENODD, which
    is as above.

    Y = DYADUP(X) is equivalent to
    Y = DYADUP(X,1,'c')
    Y = DYADUP(X,'type')  is equivalent to
    Y = DYADUP(X,1,'type')
    Y = DYADUP(X,EVENODD) is equivalent to
    Y = DYADUP(X,EVENODD,'c') 

            |1 2|                              |0 1 0 2 0|
    When X = |3 4|  we obtain:  DYADUP(X,'c') = |0 3 0 4 0|

                        |1 2|                      |1 0 2|
    DYADUP(X,'r',0) = |0 0|  , DYADUP(X,'m',0) = |0 0 0|
                        |3 4|                      |3 0 4|

    See also DYADDOWN.
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 12-Mar-96.
    Last Revision: 20-Dec-2010.
    Copyright 1995-2010 The MathWorks, Inc.
    
    Internal options.
-----------------
    Y = DYADUP(X,EVENODD,ARG) returns a vector with even length.
    Y = DYADUP([1 2 3],1,ARG) ==> [0 1 0 2 0 3]
    Y = DYADUP([1 2 3],0,ARG) ==> [1 0 2 0 3 0]
    
    Y = DYADUP(X,EVENODD,TYPE,ARG) ... for a matrix
--------------------------------------------------------------
    """

    # Special case: if input is empty, return an empty array
    if x is None or len(x) == 0:
        return np.array([])

    #x = np.asarray(x)
    if len(x.shape)<2:
        r=x.shape
        c=1
    else:
        r, c = x.shape
    rem2 = evenodd % 2
    
    if x.ndim == 1:  # Input is a vector
        if dim == 'c':
            addLEN = 0 if evenLEN else 2 * rem2 - 1
            l = 2 * len(x) + addLEN
            y = np.zeros(l, dtype=x.dtype)
            y[rem2::2] = x
        else:
            raise ValueError("Invalid dimension for vector input. Use 'c' for vector.")
        return y

    else:  # Input is a matrix
        if dim == 'c':
            nc = 2 * c + (0 if evenLEN else 2 * rem2 - 1)
            y = np.zeros((r, nc), dtype=x.dtype)
            y[:, rem2::2] = x

        elif dim == 'r':
            nr = 2 * r + (0 if evenLEN else 2 * rem2 - 1)
            y = np.zeros((nr, c), dtype=x.dtype)
            y[rem2::2, :] = x

        elif dim == 'm':
            nc = 2 * c + (0 if evenLEN else 2 * rem2 - 1)
            nr = 2 * r + (0 if evenLEN else 2 * rem2 - 1)
            y = np.zeros((nr, nc), dtype=x.dtype)
            y[rem2::2, rem2::2] = x

        else:
            raise ValueError("Invalid dimension specified. Use 'c', 'r', or 'm'.")
        
        return y
    
def wkeep1(x, length, *args):
    """
    WKEEP1  Keep part of a vector.

    Y = WKEEP1(X,L,OPT) extracts the vector Y 
    from the vector X. The length of Y is L.
    If OPT = 'c' ('l' , 'r', respectively), Y is the central
    (left, right, respectively) part of X.
    Y = WKEEP1(X,L,FIRST) returns the vector X(FIRST:FIRST+L-1).

    Y = WKEEP1(X,L) is equivalent to Y = WKEEP1(X,L,'c').
    
    MATLAB's original authors: M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 07-May-2003.
    Last Revision: 06-Feb-2011.
    Copyright 1995-2015 The MathWorks, Inc.
    """
    
    # Ensure the length is an integer and valid
    #print(length)
    if length != int(length):
        raise ValueError("Length must be an integer.")
    
    sx = len(x)
    
    # If the requested length is invalid, return the original array
    if not (0 <= length < sx):
        return x
    
    # Determine the extraction method
    if len(args) < 1:
        opt = 'c'
    else:
        opt = args[0].lower()
    
    if isinstance(opt, str):
        if opt == 'c':
            side = 0 if len(args) < 2 else args[1]
            d = (sx - length) / 2.0
            if side in ['u', 'l', '0', 0]:
                first = 1 + int(np.floor(d))
                last = sx - int(np.ceil(d))
            elif side in ['d', 'r', '1', 1]:
                first = 1 + int(np.ceil(d))
                last = sx - int(np.floor(d))
            else:
                raise ValueError("Invalid side value.")
        elif opt in ['l', 'u']:
            first = 1
            last = length
        elif opt in ['r', 'd']:
            first = sx - length + 1
            last = sx
        else:
            raise ValueError("Invalid extraction option.")
    else:
        # Assume opt is the starting index if it's not a string
        first = int(opt)
        last = first + length - 1
        
        if first < 1 or last > sx:
            raise ValueError("Invalid starting index or length.")
    
    # Convert to 0-based index for Python
    first -= 1
    last -= 1
    
    # Extract the portion of the array
    y = x[first:last+1]
    
    return y
