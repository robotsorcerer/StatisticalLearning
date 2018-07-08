function [varargout] = rfwr(action,varargin)
% rfwr implements the RFWR algorithm as suggested in
% Schaal, S., & Atkeson, C. G. (1998). Constructive incremental
% learning from only local information. Neural Comput., 10, 2047-2084.
% Depending on the keyword in the input argument "action", a certain
% number of inputs arguments will be parsed from "vargin". A variable
% number of arguments are returned according to the "action".
% See Matlab file for explanations how to use the different modalities
% of the program.
%
% Note: this implementation does not implement ridge regression. Newer
%       algorithms like LWPR and LWPPLS are much more suitable for
%       high dimensional data sets and data with ill conditioned
%       regression matrices.
%
% Copyright Stefan Schaal, March 2002

% ---------------  Different Actions of the program ------------------------

% Initialize an RFWR model:
%
% FORMAT rfwr('Init',ID, n_in, n_out, diag_only, meta, meta_rate, ...
%                    penalty, init_alpha, norm, name)
% ID              : desired ID of model
% n_in            : number of input dimensions
% n_out           : number of output dimensions
% diag_only       : 1/0 to update only the diagonal distance metric
% meta            : 1/0 to allow the use of a meta learning parameter
% meta_rate       : the meta learning rate
% penalty         : a smoothness bias, usually a pretty small number (1.e-4)
% init_alpha      : the initial learning rates
% norm            : the normalization of the inputs
% norm_out        : the normalization of the outputs
% name            : a name for the model
%
% alternatively, the function is called as
%
% FORMAT ID = rfwr('Init',ID,sdc,)
% sdc             : a complete data structure of a RFWR model
%
% returns nothing


% Change a parameter of an RFWR model:
%
% FORMAT rc = rfwr('Change',ID,pname,value)
% ID              : rfwr data structure
% pname           : name of parameter to be changed
% value           : new parameter value
%
% returns nothing


% Update an RFWR model with new data:
%
% FORMAT [yp,w] = rfwr('Update',ID,x,y)
% ID              : rfwr data structure
% x               : input data point
% y               : output data point
%
% Note: the following inputs are optional in order to use RFWR
%       in adaptive learning control with composite update laws
% e               : the tracking error of the control system
% alpha           : a strictly positive scalar to determine the
%                   magnitude of the contribution to the update
%
% returns the RFWR data structure in sdc, the prediction after
% the update, yp, and the weight of the maximally activated weight


% Predict an output for a RFWR model
%
% FORMAT [yp,w] = rfwr('Predict',ID,x)
% ID              : rfwr data structure
% x               : input data point
% cutoff          : minimal activation for prediction
%
% returns the prediction yp and the weight of the maximally activated weight


% Return the data structure of a RFWR model
%
% FORMAT [sdc] = rfwr('Structure',ID)
% ID              : rfwr data structure
%
% returns the complete data structure of a RFWR model, e.g., for saving or
% inspecting it


% Clear the data structure of a RFWR model
%
% FORMAT rfwr('Clear',ID)
% ID              : rfwr data structure
%
% returns nothing

% the structure storing all RFWR models
global sdcs;


if nargin < 2,
  error('Incorrect call to rfwr');
end

switch action,

  %..............................................................................
  % Initialize a new RFWR model
  case 'Init'

    % check whether a complete model was
    % given or data for a new model

    if (nargin == 3)

      ID       = varargin{1};
      sdcs(ID) = varargin{2};

    else

      % copy from input arguments
      ID                  = varargin{1};
      sdcs(ID).n_in       = varargin{2};
      sdcs(ID).n_out      = varargin{3};
      sdcs(ID).diag_only  = varargin{4};
      sdcs(ID).meta       = varargin{5};
      sdcs(ID).meta_rate  = varargin{6};
      sdcs(ID).penalty    = varargin{7};
      sdcs(ID).init_alpha = varargin{8};
      sdcs(ID).norm       = varargin{9};
      sdcs(ID).norm_out   = varargin{10};
      sdcs(ID).name       = varargin{11};

      % add additional convenient variables
      sdcs(ID).n_data       = 0;
      sdcs(ID).w_gen        = 0.1;
      sdcs(ID).w_prune      = 0.9;
      sdcs(ID).init_lambda  = 0.999;
      sdcs(ID).final_lambda = 0.9999;
      sdcs(ID).tau_lambda   = 0.99999;
      sdcs(ID).init_P       = 1.e+10;
      sdcs(ID).n_pruned     = 0;

      % other variables
      sdcs(ID).init_D       = eye(sdcs(ID).n_in)*25;
      sdcs(ID).init_M       = chol(sdcs(ID).init_D);
      sdcs(ID).init_alpha   = ones(sdcs(ID).n_in)*sdcs(ID).init_alpha;
      sdcs(ID).mean_x       = zeros(sdcs(ID).n_in,1);
      sdcs(ID).var_x        = zeros(sdcs(ID).n_in,1);
      sdcs(ID).rfs          = [];
      sdcs(ID).kernel       = 'Gaussian';

    end


    %..............................................................................
  case 'Change'

    ID = varargin{1};
    command = sprintf('sdcs(%d).%s = varargin{3};',ID,varargin{2});
    eval(command);

    % make sure some initializations remain correct
    sdcs(ID).init_M       = chol(sdcs(ID).init_D);


    %..............................................................................
  case 'Update'
    ID  = varargin{1};
    x   = varargin{2};
    y   = varargin{3};

    if (nargin > 4)
      composite_control = 1;
      e_t   = varargin{4};
      alpha = varargin{5};
    else
      composite_control = 0;
    end

    % update the global mean and variance of the training data for
    % information purposes
    sdcs(ID).mean_x = (sdcs(ID).mean_x*sdcs(ID).n_data + x)/(sdcs(ID).n_data+1);
    sdcs(ID).var_x  = (sdcs(ID).var_x*sdcs(ID).n_data + (x-sdcs(ID).mean_x).^2)/(sdcs(ID).n_data+1);
    sdcs(ID).n_data = sdcs(ID).n_data+1;

    % normalize the inputs
    xn = x./sdcs(ID).norm;

    % normalize the outputs
    yn = y./sdcs(ID).norm_out;

    % check all RFs for updating
    % wv is a vector of 3 weights, ordered [w; sec_w; max_w]
    % iv is the corresponding vector containing the RF indices
    wv = zeros(3,1);
    iv = zeros(3,1);
    yp = zeros(size(y));
    sum_w = 0;
    tms = zeros(length(sdcs(ID).rfs));

    for i=1:length(sdcs(ID).rfs),

      % compute the weight and keep the three larget weights sorted
      w  = compute_weight(sdcs(ID).diag_only,sdcs(ID).kernel,sdcs(ID).rfs(i).c,sdcs(ID).rfs(i).D,xn);
      sdcs(ID).rfs(i).w = w;
      wv(1) = w;
      iv(1) = i;
      [wv,ind]=sort(wv);
      iv = iv(ind);

      % only update if activation is high enough
      if (w > 0.001),

        rf = sdcs(ID).rfs(i);

        % update weighted mean for xn and y, and create mean-zero
        % variables
        [rf,xmz,ymz] = update_means(sdcs(ID).rfs(i),xn,yn,w);

        % update the regression
        [rf,yp_i,e_cv,e] = update_regression(rf,xmz,ymz,w);
        if (rf.trustworthy),
          yp = w*yp_i + yp;
          sum_w = sum_w + w;
        end

        % update the distance metric
        [rf,tm] = update_distance_metric(ID,rf,xmz,ymz,w,e_cv,e,xn);
        tms(i) = 1;

        % update simple statistical variables
        rf.sum_w  = rf.sum_w*rf.lambda + w;
        rf.n_data = rf.n_data*rf.lambda + 1;
        rf.lambda = sdcs(ID).tau_lambda * rf.lambda + sdcs(ID).final_lambda*(1.-sdcs(ID).tau_lambda);

        % incorporate updates
        sdcs(ID).rfs(i) = rf;

      else

        sdcs(ID).rfs(i).w = 0;

      end % if (w > 0.001)

    end


    % if RFWR is used for control, incorporate the tracking error
    if (composite_control),
      inds = find(tms > 0);
      if ~isempty(inds),
        for j=1:length(inds),
          i = inds(j);
          sdcs(ID).rfs(i).B  = sdcs(ID).rfs(i).B  + alpha * tms(j) * sdcs(ID).rfs(i).P     * sdcs(ID).rfs(i).w/sum_w * (xn-sdcs(ID).rfs(i).c) * e_t;
          sdcs(ID).rfs(i).b0 = sdcs(ID).rfs(i).b0 + alpha * tms(j) / sdcs(ID).rfs(i).sum_w * sdcs(ID).rfs(i).w/sum_w * e_t;
        end
      end
    end


    % do we need to add a new RF?
    if (wv(3) <= sdcs(ID).w_gen),
      if (wv(3) > 0.1*sdcs(ID).w_gen & sdcs(ID).rfs(iv(3)).trustworthy),
        sdcs(ID).rfs(length(sdcs(ID).rfs)+1)=init_rf(ID,sdcs(ID).rfs(iv(3)),xn,yn);
      else
        if (length(sdcs(ID).rfs)==0),
          sdcs(ID).rfs = init_rf(ID,[],xn,y);
        else
          sdcs(ID).rfs(length(sdcs(ID).rfs)+1) = init_rf(ID,[],xn,yn);
        end
      end
    end

    % do we need to prune a RF? Prune the one with smaller D
    if (wv(2:3) > sdcs(ID).w_prune),
      if (sum(sum(sdcs(ID).rfs(iv(2)).D)) > sum(sum(sdcs(ID).rfs(iv(3)).D)))
        sdcs(ID).rfs(iv(2)) = [];
        disp(sprintf('%d: Pruned #RF=%d',ID,iv(2)));
      else
        sdcs(ID).rfs(iv(3)) = [];
        disp(sprintf('%d: Pruned #RF=%d',ID,iv(3)));
      end
      sdcs(ID).n_pruned = sdcs(ID).n_pruned + 1;
    end

    % the final prediction
    if (sum_w > 0),
      yp = yp.*sdcs(ID).norm_out/sum_w;
    end

    varargout(1) = {yp};
    varargout(2) = {wv(3)};


    %..............................................................................
  case 'Predict'
    ID     = varargin{1};
    x      = varargin{2};
    cutoff = varargin{3};

    % normalize the inputs
    xn = x./sdcs(ID).norm;

    % maintain the maximal activation
    max_w = 0;
    yp = zeros(sdcs(ID).n_out,1);
    sum_w = 0;

    for i=1:length(sdcs(ID).rfs),

      % compute the weight
      w  = compute_weight(sdcs(ID).diag_only,sdcs(ID).kernel,sdcs(ID).rfs(i).c,sdcs(ID).rfs(i).D,xn);
      sdcs(ID).rfs(i).w = w;
      max_w = max([max_w,w]);

      % only predict if activation is high enough
      if (w > cutoff & sdcs(ID).rfs(i).trustworthy),

        % the mean zero input
        xmz = xn - sdcs(ID).rfs(i).mean_x;

        % the prediction
        yp = yp + (sdcs(ID).rfs(i).B'*xmz + sdcs(ID).rfs(i).b0) * w;
        sum_w = sum_w + w;

      end % if (w > cutoff)

    end

    % the final prediction
    if (sum_w > 0),
      yp = yp.*sdcs(ID).norm_out/sum_w;
    end

    varargout(1) = {yp};
    varargout(2) = {max_w};


    %..............................................................................
  case 'Structure'
    ID     = varargin{1};

    varargout(1) = {sdcs(ID)};


    %..............................................................................
  case 'Clear'
    ID     = varargin{1};

    sdcs(ID) = [];

end


%-----------------------------------------------------------------------------
function rf=init_rf(ID,template_rf,c,y)
% initialize a local model

global sdcs;

if ~isempty(template_rf),
  rf = template_rf;
else
  rf.D     = sdcs(ID).init_D;
  rf.M     = sdcs(ID).init_M;
  rf.alpha = sdcs(ID).init_alpha;
  rf.b0    = y;                             % the weighted mean of output
end

rf.P           = eye(sdcs(ID).n_in)*sdcs(ID).init_P;  % the inverse covariance matrix
rf.B           = zeros(sdcs(ID).n_in,sdcs(ID).n_out); % the regression parameters
rf.H           = zeros(sdcs(ID).n_in,sdcs(ID).n_out); % a memory term for sufficient stats
rf.R           = zeros(sdcs(ID).n_in,sdcs(ID).n_in);  % a memory term for sufficient stats
rf.h           = zeros(size(rf.alpha));     % a memory term for 2nd order gradients
rf.b           = log(rf.alpha+1.e-10);      % a memory term for 2nd order gradients
rf.c           = c;                         % the center of the RF
rf.vif         = 1.e10*ones(sdcs(ID).n_in,1);    % the variance inflation factor
rf.sum_w       = 0;                         % the sum of weights
rf.sum_e_cv2   = 0;                         % weighted sum of cross.valid. err.
rf.sum_e2      = 0;                         % weighted sum of error (not CV)
rf.n_data      = 0;                         % discounted amount of data in RF
rf.trustworthy = 0;                         % indicates statistical confidence
rf.lambda      = sdcs(ID).init_lambda;
rf.mean_x      = 0;                         % the weighted mean of inputs
rf.var_x       = 0;                         % the weighted variance of inputs
rf.w           = 0;                         % store the last computed weight


%-----------------------------------------------------------------------------
function w=compute_weight(diag_only,kernel,c,D,x)
% compute the weight

% subtract the center
x = x-c;

if diag_only,
  d2 = x'*(diag(D).*x);
else,
  d2 = x'*D*x;
end

switch kernel
  case 'Gaussian'
    w = exp(-0.5*d2);
  case 'BiSquare'
    if (0.5*d2 > 1)
      w = 0;
    else
      w = (1-0.5*d2)^2;
    end
end


%-----------------------------------------------------------------------------
function [rf,xmz,ymz]=update_means(rf,x,y,w)
% update means and computer mean zero variables

rf.mean_x = (rf.sum_w*rf.mean_x*rf.lambda + w*x)/(rf.sum_w*rf.lambda+w);
rf.var_x  = (rf.sum_w*rf.var_x*rf.lambda + w*(x-rf.mean_x).^2)/(rf.sum_w*rf.lambda+w);
rf.b0     = (rf.sum_w*rf.b0*rf.lambda + w*y)/(rf.sum_w*rf.lambda+w);
xmz = x - rf.mean_x;
ymz = y - rf.b0;


%-----------------------------------------------------------------------------
function [rf,yp,e_cv,e] = update_regression(rf,x,y,w)
% update the linear regression parameters

% update the model
Px     = rf.P*x;
xP     = Px';
e_cv   = y - rf.B'*x;
P      = (rf.P - Px*xP/(rf.lambda/w + xP*x))/rf.lambda;
B      = rf.B + w * (P * x) * e_cv';

% update the RF
rf.B = B;
rf.P = P;

% the new predicted output after updating
yp = rf.B'*x;
e  = y - rf.B'*x;
yp = yp + rf.b0;

% is the RF trustworthy: use variance inflation factor to judge
rf.vif = diag(P).*rf.var_x;
if (max(rf.vif) < 10 & rf.n_data > length(x)*2)
  rf.trustworthy = 1;
end


%-----------------------------------------------------------------------------
function [rf,transient_multiplier] = update_distance_metric(ID,rf,x,y,w,e_cv,e,xn)

global sdcs;

% update the distance metric

if (rf.vif > 5)
  transient_multiplier = 0;
  return;
end

penalty   = sdcs(ID).penalty/length(x); % normalizes penality w.r.t. number of inputs
meta      = sdcs(ID).meta;
meta_rate = sdcs(ID).meta_rate;
kernel    = sdcs(ID).kernel;
diag_only = sdcs(ID).diag_only;

% useful pre-computations: they need to come before the updates
e_cv2                = e_cv'*e_cv;
e2                   = e'*e;
h                    = w*x'*rf.P*x;
W                    = rf.sum_w*rf.lambda + w;
rf.sum_e_cv2         = rf.sum_e_cv2*rf.lambda + w*e_cv2;
rf.sum_e2            = rf.sum_e2*rf.lambda + w*e2;
E                    = rf.sum_e_cv2;
transient_multiplier = (rf.sum_e2/(rf.sum_e_cv2+1.e-10))^4; % this is a numerical safety heuristic
n_out                = length(y);

% the derivative dJ1/dw
Px    = rf.P*x;
Pxe   = Px*e';
dJ1dw = -E/W^2 + 1/W*(e_cv2 - sum(sum((2*Pxe).*rf.H)) - sum(sum((2*Px*Px').*rf.R)));

% the derivatives dw/dM and dJ2/dM
[dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM] = dist_derivatives(w,rf,xn-rf.c,diag_only,kernel,penalty,meta);

% the final derivative becomes (note this is upper triangular)
dJdM = dwdM*dJ1dw/n_out + w/W*dJ2dM;

% the second derivative if meta learning is required, and meta learning update
if (meta)

  % second derivatives
  dJ1J1dwdw = -e_cv2/W^2 - 2/W*sum(sum((-Pxe/W -2*Px*(x'*Pxe)).*rf.H)) + 2/W*e2*h/w - ...
    1/W^2*(e_cv2-2*sum(sum(Pxe.*rf.H))) + 2*E/W^3;

  dJJdMdM = (dwwdMdM*dJ1dw + dwdM.^2*dJ1J1dwdw)/n_out + w/W*dJ2J2dMdM;

  % update the learning rates
  aux = meta_rate * transient_multiplier * (dJdM.*rf.h);

  % limit the update rate
  ind = find(abs(aux) > 0.1);
  if (~isempty(ind)),
    aux(ind) = 0.1*sign(aux(ind));
  end
  rf.b = rf.b - aux;

  % prevent numerical overflow
  ind = find(abs(rf.b) > 10);
  if (~isempty(ind)),
    rf.b(ind) = 10*sign(rf.b(ind));
  end

  rf.alpha = exp(rf.b);

  aux = 1 - (rf.alpha.*dJJdMdM) * transient_multiplier ;
  ind = find(aux < 0);
  if (~isempty(ind)),
    aux(ind) = 0;
  end

  rf.h = rf.h.*aux - (rf.alpha.*dJdM) * transient_multiplier;

end

% update the distance metric, use some caution for too large gradients
maxM = max(max(abs(rf.M)));
delta_M = rf.alpha.*dJdM*transient_multiplier;
ind = find(delta_M > 0.1*maxM);
if (~isempty(ind)),
  rf.alpha(ind) = rf.alpha(ind)/2;
  delta_M(ind) = 0;
  disp(sprintf('Reduced learning rate'));
end
rf.M = rf.M - rf.alpha.*dJdM*transient_multiplier;
rf.D = rf.M'*rf.M;


% update sufficient statistics: note this must come after the updates
rf.H = rf.lambda*rf.H + (w/(1-h))*x*e_cv'*transient_multiplier;
rf.R = rf.lambda*rf.R + (w^2*e_cv2/(1-h))*(x*x')*transient_multiplier;


%-----------------------------------------------------------------------------
function [dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM] = dist_derivatives(w,rf,dx,diag_only,kernel,penalty,meta)
% compute derivatives of distance metric: note that these will be upper
% triangular matrices for efficiency

n_in      = length(dx);
dwdM      = zeros(n_in);
dJ2dM     = zeros(n_in);
dJ2J2dMdM = zeros(n_in);
dwwdMdM   = zeros(n_in);

for n=1:n_in,
  for m=n:n_in,

    sum_aux    = 0;
    sum_aux1   = 0;

    % take the derivative of D with respect to nm_th element of M */

    if (diag_only & n==m),

      aux = 2*rf.M(n,n);
      dwdM(n,n) = dx(n)^2 * aux;
      sum_aux = rf.D(n,n)*aux;
      if (meta)
        sum_aux1 = sum_aux1 + aux^2;
      end

    elseif (~diag_only),

      for i=n:n_in,

        % aux corresponds to the in_th (= ni_th) element of dDdm_nm
        % this is directly processed for dwdM and dJ2dM

        if (i == m)
          aux = 2*rf.M(n,i);
          dwdM(n,m) = dwdM(n,m) + dx(i) * dx(m) * aux;
          sum_aux = sum_aux + rf.D(i,m)*aux;
          if (meta)
            sum_aux1 = sum_aux1 + aux^2;
          end
        else
          aux = rf.M(n,i);
          dwdM(n,m) = dwdM(n,m) + 2. * dx(i) * dx(m) * aux;
          sum_aux = sum_aux + 2.*rf.D(i,m)*aux;
          if (meta)
            sum_aux1 = sum_aux1 + 2*aux^2;
          end
        end

      end

    end

    switch kernel
      case 'Gaussian'
        dwdM(n,m)  = -0.5*w*dwdM(n,m);
      case 'BiSquare'
        dwdM(n,m)  = -sqrt(w)*dwdM(n,m);
    end

    dJ2dM(n,m)  = 2.*penalty*sum_aux;

    if (meta)
      dJ2J2dMdM(n,m) = 2.*penalty*(2*rf.D(m,m) + sum_aux1);
      dJ2J2dMdM(m,n) = dJ2J2dMdM(n,m);
      switch kernel
        case 'Gaussian'
          dwwdMdM(n,m)   = dwdM(n,m)^2/w - w*dx(m)^2;
        case 'BiSquare'
          dwwdMdM(n,m)   = dwdM(n,m)^2/w/2 - 2*sqrt(w)*dx(m)^2;
      end
      dwwdMdM(m,n)   = dwwdMdM(n,m);
    end

  end
end

