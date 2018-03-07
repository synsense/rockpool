function mfWHat = RandomEIUnitaryEigenvalueNet(nNumExc, nNumInh, fInhTauFactor, fInhWFactor, fhRand)

% - Check arguments
if ~exist('fInhTauFactor', 'var') || isempty(fInhTauFactor)
   fInhTauFactor = 1;
end

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

vfD(real(vfD) > 1-eps) = 0;

mfW = real(V * diag(vfD) / V);

% - Compute Jacobian
mfJ = mfW - eye(size(mfW));
mfJ(vnInh, :) = mfJ(vnInh, :) ./ fInhTauFactor;

% - Check for unstable eigenvectors
[V, D] = eig(mfJ);

% - Identify and wipe unstable eigenvectors
vbUnstablePartition = all(V > 0, 1);
D(vbUnstablePartition, vbUnstablePartition) = 0;
% fScale = max(real(diag(D(~vbUnstablePartition, ~vbUnstablePartition))));

% - Reconstruct weight matrix
mfJHat = real(V * D / V);
mfWHat = mfJHat;
mfWHat(vnInh, :) = mfJHat(vnInh, :) .* fInhTauFactor;
mfWHat = mfWHat + eye(size(mfWHat));
mfWHat = mfWHat;% ./ fScale;

