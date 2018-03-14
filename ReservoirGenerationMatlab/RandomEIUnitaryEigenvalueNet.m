function mfW = RandomEIUnitaryEigenvalueNet(nNumExc, nNumInh, fInhWFactor, fhRand)

% - Check arguments
if ~exist('fhRand', 'var') || isempty(fhRand)
   fhRand = @(N)randn(N)./sqrt(N);
end

mfW = fhRand(nNumExc + nNumInh);
vnExc = 1:nNumExc;
mfWE = mfW(:, vnExc);
mfWE(mfWE < 0) = 0;

vnInh = nNumExc + (1:nNumInh);
mfWI = mfW(:, vnInh) * fInhWFactor;
mfWI(mfWI > 0) = 0;

mfW = [mfWE mfWI];

% - Set eigenstructure directly
[V, D] = eig(mfW);
vfD = diag(D) ./ abs(diag(D));

vfD = complex(abs(real(vfD)), sign(imag(vfD)));

vfD(real(vfD) > 1-eps) = -1;

mfW = real(V * diag(vfD) / V);