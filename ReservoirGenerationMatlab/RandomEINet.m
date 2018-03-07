function [mfWHat, mfJHat] = RandomEINet(nNumExc, nNumInh, fInhTauFactor, fInhWFactor, fhRand)

% - Check arguments
if ~exist('fInhTau', 'var') || isempty(fInhTauFactor)
   fInhTauFactor = 1;
end

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

% - Compute Jacobian
mfJ = mfW - eye(size(mfW));
mfJ(vnInh, :) = mfJ(vnInh, :) ./ fInhTauFactor;

% - Check for unstable eigenvectors
[V, D] = eig(mfJ);

% - Identify and wipe unstable eigenvectors
vbUnstablePartition = all(V > 0, 1);
D(vbUnstablePartition, vbUnstablePartition) = 0;

% - Reconstruct weight matrix
mfJHat = real(V * D / V);
mfWHat = mfJHat;
mfWHat(vnInh, :) = mfJHat(vnInh, :) .* fInhTauFactor;
mfWHat = mfWHat + eye(size(mfWHat));
mfWHat = mfWHat .* fInhTauFactor * 30;

