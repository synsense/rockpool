function [t, mfX, vfEigs] = SimulateLT(mfW, vfBias, vfInput, tLimit, vfTau, fZeta, tNoiseFrameDur)

% -- Check arguments
nNetSize = size(mfW, 1);

% - Handle static or functional inputs
if isa(vfInput, 'function_handle')
   fhInput = vfInput;
else
   fhInput = @(t)vfInput;
end

if numel(fhInput(0)) == 1
   fhInputOrig = fhInput;
   fhInput = @(t)repmat(fhInputOrig(t), 1, nNetSize);
end

vfX = fhInput(0);

if (numel(vfX) ~= nNetSize) && ~(numel(vfX) == 1)
   error('N must be equal for network and ''vfInput''.');
end

if ~exist('vfBias', 'var') || isempty(vfBias)
   vfBias = zeros(nNetSize, 1);
end

if (numel(vfBias) ~= nNetSize) && ~(numel(vfBias) == 1)
   error('N must be equal for network and ''vfBias''.');
end

if ~exist('vfTau', 'var') || isempty(vfTau)
   vfTau = ones(nNetSize, 1);
end

if (numel(vfTau) ~= nNetSize) && ~(numel(vfTau) == 1)
   error('N must be equal for network and ''vfTau''.');
end

% - Generate noise
if exist('fZeta', 'var') && ~isempty(fZeta)
   if ~exist('tNoiseFrameDur', 'var') || isempty(tNoiseFrameDur)
      tNoiseFrameDur = 1;
   end
   
   % - Build a time trace for the noise
   vtNoiseTimeTrace = 0:tNoiseFrameDur:tLimit;
   
   % - Make sure the noise trace spans the stimulus
   if vtNoiseTimeTrace(end) < tLimit
      vtNoiseTimeTrace(end+1) = tLimit;
   end
   
   mfNoiseData = normrnd(0, fZeta ./ sqrt(tNoiseFrameDur), numel(vtNoiseTimeTrace), nNetSize);
   fhNoiseFunc = @(t)interp1(vtNoiseTimeTrace, mfNoiseData, t);
else
   fhNoiseFunc = @(t)0;
end


% - Define linear-threhsold dynamical system
XdTRec = @(vfX, Wrec, vfBias, vfInput, vfNoise, vfTau)(Wrec * max(vfX(:) + vfBias(:), 0) + vfInput(:) + vfNoise(:) - vfX(:)) ./ vfTau(:);

[t, mfX] = ode45(@(t, x)XdTRec(x, mfW, vfBias, fhInput(t), fhNoiseFunc(t), vfTau), [0 tLimit], vfX);

if nargout > 2
   mfJ = mfW - eye(nNetSize);
   mfJ = bsxfun(@rdivide, mfJ, vfTau(:));
   vfEigs = complex(eig(mfJ));
end

end