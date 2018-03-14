function mfW = RandomEINet(nNumExc, nNumInh, fInhWFactor, fhRand)

if ~exist('fInhWFactor', 'var') || isempty(fInhWFactor)
   fInhWFactor = 1;
end
   
if ~exist('fhRand', 'var') || isempty(fhRand)
   fhRand = @(N)randn(N)./sqrt(N);
end

% - Generate random connectivity matrix
mfW = fhRand(nNumExc + nNumInh);
vnExc = 1:nNumExc;
mfWE = mfW(:, vnExc);
mfWE(mfWE < 0) = 0;
mfWE = bsxfun(@rdivide, mfWE, sum(mfWE));

vnInh = nNumExc + (1:nNumInh);
mfWI = mfW(:, vnInh);
mfWI(mfWI > 0) = 0;
mfWI = -bsxfun(@rdivide, mfWI, sum(mfWI)) * abs(fInhWFactor);

mfW = [mfWE mfWI];

