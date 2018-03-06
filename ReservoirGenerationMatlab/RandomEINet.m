function mfW = RandomEINet(nNumExc, nNumInh, fhRand)

if ~exist('fhRand', 'var') || isempty(fhRand)
   fhRand = @(N)randn(N)./sqrt(N);
end

mfW = fhRand(nNumExc + nNumInh);
mfWE = mfW(:, 1:nNumExc);
mfWE(mfWE < 0) = 0;
mfWE = bsxfun(@rdivide, mfWE, sum(mfWE));

mfWI = mfW(:, nNumExc + (1:nNumInh));
mfWI(mfWI > 0) = 0;
mfWI = -bsxfun(@rdivide, mfWI, sum(mfWI));

mfW = [mfWE mfWI];