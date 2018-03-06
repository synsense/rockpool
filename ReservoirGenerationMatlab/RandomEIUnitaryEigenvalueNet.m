function mfW = RandomEIUnitaryEigenvalueNet(nNumExc, nNumInh, fhRand)

if ~exist('fhRand', 'var') || isempty(fhRand)
   fhRand = @(N)randn(N)./sqrt(N);
end

mfW = fhRand(nNumExc + nNumInh);
mfWE = mfW(:, 1:nNumExc);
mfWE(mfWE < 0) = 0;

mfWI = mfW(:, nNumExc + (1:nNumInh));
mfWI(mfWI > 0) = 0;

mfW = [mfWE mfWI];

[V, D] = eig(mfW);
vfD = diag(D) ./ abs(diag(D));

vfD = complex(abs(real(vfD)), sign(imag(vfD)));

vfD(real(vfD) > 1-eps) = 0;

mfW = real(V * diag(vfD) / V);