classdef RocketLander_6dof < rl.env.MATLABEnvironment
% ROCKETLANDER environment models a 6-DOF disc-shaped rocket with mass. 
% The rocket has 4 thrusters for forward and rotational motion like a Quadcopter. 
% In this version Gravity acts vertically downwards, and there are no aerodynamic drag forces.
% Revised: 1400-06-23
    properties
  %% Variables & Parameters Declaration
           % Parameters
           
            L1 = 10;% C.G. to top/bottom end (m)
            L2 =  5; % C.G. to left/right end (m)     
            Gravity = 9.806;% Acceleration due to gravity (m/s^2)
            ThrustLimits = [10 10 10 10];% Bounds on thrust (N)
            Ts = 0.1;% Sample time (s)
            State = zeros(12,1);% State vector     
            LastAction = zeros(4,1);% Last Action values
            LastShaping = 0;% Last reward value for distance and speed
            DistanceIntegral = 0
            VelocityIntegral = 0
            TimeCount = 0;% Time elapsed during simulation (sec)

%             Deltat = 0.01;       % Value of discretization of the time interval (s)
            Mass = 0.45;            % Mass of drone (Kg)
            l = 0.23;            % Length of drone arms, from the center (m)
             L = 3e-6;            % Lift constant
%             A = 0.25/m*eye(3);   % Aerodynamical effects matrix
%             cost = l*L;

            %Ir = 6e-5;  % inertia for a flat disk:I = 0.5*m*L1_^2;
         
           %    Variables
            % w is the matrix of angular velocities of the 4 motors x Time interval
            % (rad/s)
            w = zeros(4,1);
            %   Initial Conditions, supposedly zero; Preallocation
            xi = zeros(3, 2);   % Relative position (x,y,z) of drone, relative to Earth Frame
            x_dot = zeros(3,2);   % Derivate of xi
            eta = zeros(3, 2);  % Attitudes of drone (roll, pitch, yaw)
            tau = zeros(3,1);   % Torques of drone, relative to body frame attitudes
            T = 0;              % Thrust of drone on the z axis
            vel = zeros(3, 2);  % Angular velocities relative to the Body Frame
            Winv = zeros(3,3);  % Tranformation matrix (inverted)
            R = zeros(3,1);     % Rotation vector of drone's z axis
            Rx = zeros(3,3,2);  % x Rotation matrix
            Ry = zeros(3,3,2);  % y Rotation matrix
            Rz = zeros(3,3,2);  % z Rotation matrix
        
    end
    
    properties (Hidden)        
        % Agent is in continuous mode
        UseContinuousActions = false 
        
        % Log for actions and states
        LoggedSignals = cell(4,1)
        
        % Flags for visualization
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties (Transient,Access = private)
        d = []
    end
    
    methods
        
        function this = RocketLander_6dof(ActionInfo)
            
            % Define observation info
            ObservationInfo(1) = rlNumericSpec([13 1]);
            ObservationInfo(1).Name = 'states';
           
            % Define action info
            ActionInfo(1) = rlNumericSpec([4 1]);
            ActionInfo(1).Name = 'thrusts';
           
            % Create environment
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Update action info and initialize states and logs
            updateActionInfo(this);
            this.State = [0 0 this.L1 0 0 0 0 0 0 0 0 0]';
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0 0 0]';
            
        end
        
        function set.UseContinuousActions(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','UseContinuousActions');
            this.UseContinuousActions = logical(val);
            updateActionInfo(this);
        end
        
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',12},'','State');% 12
            this.State = state(:);
            notifyEnvUpdated(this);
        end
        
        function set.Mass(this,val)

            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Mass');
            this.Mass = val;
        end
        
        function set.L1(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L1');
            this.L1 = val;
        end
        
        function set.L2(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L2');
            this.L2 = val;
        end
%         function set.L3(this,val)
%             validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L3');
%             this.L3 = val;
%         end
        
        function set.ThrustLimits(this,val)
            validateattributes(val,{'numeric'},{'finite','real','vector','numel',4},'','ThrustLimits');% 4
            this.ThrustLimits = sort(val);
        end
        
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','Gravity');
            this.Gravity = val;
            updateActionInfo(this);
        end
        
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        
        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = 	(this);% 3D Plot
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end
        
        function [nextobs,reward,isdone,loggedSignals] = step(this,Action)
            
            loggedSignals = [];
            
            % Scale up the actions
            action = Action .* this.ThrustLimits(2);
            
            % Trapezoidal integration
            ts = this.Ts;
            x1 = this.State(1:12);%12 
            [dx1,~,~,action] = dynamics(this,x1, action);
            dx2 = dynamics(this,x1 + ts*dx1, action);
            x = ts/2*(dx1 + dx2) + x1;
            x(5) = atan2(sin(x(3)),cos(x(3)));   % wrap the output angle
            
            % Update time count
            this.TimeCount = this.TimeCount + this.Ts;
            
            % Unpack states
            x_  = x(1);
            y_  = x(2);
            z_  = x(3) - this.L1; % teta
            
            phi_ = x(4);
            teta_ = x(5);
            si_ = x(6);
            
            dx_ = x(7);
            dy_ = x(8);
            dz_ = x(9); % #ok<NASGU>

            dphi_ = x(10);
            dteta_ = x(11);
            dsi_ =  x(12);
            % Determine conditions
            bounds = [100 120-this.L1 120 60 60 60  pi pi pi  pi/2 pi/2 pi/2];  % bounds on the state-space
            % [x y z x_dot y_dot z_dot phi teta si phi_dot teta_dot si_dot]
            isOutOfBounds = any(abs(x) > bounds(:));
            collision = z_ <= 0;
            roughCollision = collision && (dz_ < -0.5 || abs(dx_) > 0.5);%%
            softCollision  = collision && (dz_ >= -0.5 && abs(dx_) <= 0.5);%%            

            % Shape the reward
            distance = sqrt(x_^2 + y_^2 + z_^2) / sqrt(100^2+120^2+100^2);% z_^2
            speed = sqrt(dx_^2 + dy_^2 + dz_^2) / sqrt(60^2+60^2+60^2);% dz_^2
            s = 1 - (sqrt(distance) + 0.5*sqrt(speed));
            shaping = s - this.LastShaping;
            reward = shaping - 0.1*t_^2 - 0.01*sum(Action.^2);
            reward = reward + 500 * softCollision;
            this.LastShaping = shaping;
            
            % Set the states and last action
            this.State = x;
            this.LastAction = Action(:);
            
            % Set the observations
            xhat = x(1)/bounds(1);  % Normalize to [-1,1]
            yhat = x(2)/bounds(2);
            zhat = x(3)/bounds(3); 
            dxhat = x(4)/bounds(4); 
            dyhat = x(5)/bounds(5);
            dzhat = x(6)/bounds(6);
            
            
            phihat  = x(7)/bounds(7);
            tetahat = x(8)/bounds(8);
            sihat   = x(9)/bounds(9);
            dphihat = x(10)/bounds(10);
            dtetahat= x(12)/bounds(11);
            dsihat  = x(12)/bounds(12);
            
            landingSensor = 0;
            if roughCollision
                landingSensor = -1;
            elseif softCollision
                landingSensor = 1;
            else
                landingSensor = 0;
            end
            nextobs = [xhat; yhat; zhat; dxhat; dyhat; dzhat;phihat; tetahat; sihat; dphihat; dtetahat; dsihat; landingSensor];
            
            % Log states and actions
            this.LoggedSignals(1) = {[this.LoggedSignals{1}, this.State(:)]};
            this.LoggedSignals(2) = {[this.LoggedSignals{2}, action(:)]};
            
            % Terminate
            isdone = isOutOfBounds || collision;
        end
        
        function obs = reset(this)
            x0 = 0;
            y0 = 0;
            z0 = 100;
            
            phi0 = 0;
            teta0 = 0;
            si0 = 0;
            
            if rand
                x0 = -20 + 40*rand;             % vary x from [-20,+20] m
                y0 = -20 + 40*rand;             % vary y from [-20,+20] m

                phi0 = pi/180 * (-45 + 90*rand);  % vary phi from [-45,+45] deg
                teta0 = pi/180 * (-45 + 90*rand);  % vary teta from [-45,+45] deg
                si0 = pi/180 * (-45 + 90*rand);  % vary si from [-45,+45] deg
            end
            x = [x0; y0; z0; phi0; teta0; si0; 0; 0; 0; 0; 0; 0];
            
            this.State = x;
            obs = [x; 0];
            this.TimeCount = 0;
            this.LastShaping = 0;
            this.LoggedSignals{1} = this.State;
            this.LoggedSignals{2} = [0 0 0 0]';     
        end
        
    end
    
    methods (Access = private)
        
        function updateActionInfo(this)
            LL = 0; 
            UL = 1;
            if this.UseContinuousActions
                this.ActionInfo(1) = rlNumericSpec([4 1 1],'LowerLimit',LL,'UpperLimit',UL);% 4
            else
                ML = (UL - LL)/2 + LL;
                els = {...
                    [LL;LL],...  % do nothing
                    [LL;ML],...  % fire right (med)  
                    [LL;UL],...  % fire right (max)
                    [ML;LL],...  % fire left (med)
                    [ML;ML],...  % fire left (med) + right (med)
                    [ML;UL],...  % fire left (med) + right (max)
                    [UL;LL],...  % fire left (max)
                    [UL;ML],...  % fire left (max) + right (med)
                    [UL;UL] ...  % fire left (max) + right (max)
                    }';
                this.ActionInfo = rlFiniteSetSpec(els);
            end
            this.ActionInfo(1).Name = 'thrusts';
        end
        
     function [dx,T1,T2,action] = dynamics(this,x,action)
%          
        % DYNAMICS calculates the state derivatives of the robot.
   
            
            action = max(this.ThrustLimits(1),min(this.ThrustLimits(4),action));
            
            T1 = action(1);
            T2 = action(2);
            T3 = action(3);
            T4 = action(4);
            
            
            
            L1_ = this.L1;
            L2_ = this.L2;
            m = this.Mass;
            g = this.Gravity;
           
%%
            Ixx = 5e-3;
            Iyy = 5e-3;
            Izz = 8e-3;
            I = [Ixx 0 0;
                 0 Iyy 0;
                 0 0 Izz];
            b = 7.5e-7 ;       % Thrust constant
            drag = 0.01;       % Drag constant
            
            u1 = b.*(T1+T2+T3+T4);
            u2 = b.l .*(T2 - T4);
            u3 = b.l .*(T1 - T3);
            u4 = drag .* (T2 + T4 - T1 - T3);

            %%
%             x_   = x(1); %#ok<NASGU>
%             y_   = x(2);
%             z_   = x(3); 
%             
%             dx_ = x(4);
%             dy_ = x(5);
%             dz_ = x(6); 
%             
%             
%             phi_ = x(7);           
%             teta_= x(8);
%             si_  = x(9);
%             dphi_ = x(10);
%             dteta_ = x(11);
%             dsi_   = x(12);

            dx = zeros(12,1);
            
            % ground penetration
            zhat = z_ - L1_;
            

            dx(1) = dx_;
            dx(2) = dy_;
            dx(3) = dz_;
            dx(4) = -g * teta_;
            dx(5) = g * phi_;
            dx(6) = -u1/m;
            
            dx(7) = dphi_ ;
            dx(8) = dteta_ ;
            dx(9) = dsi_;
            dx(10) = u2 / Ixx;
            dx(11) = u3 / Iyy;
            dx(12) = u4 / Izz;
            
            
            if zhat < 0
                % "normalized" for mass
                k = 1e2;
                c = 1e1;
                % treat as rolling wheel (1 DOF is lost between x and theta)
                dx(4) = (Fx*L1_^2 - Mz*L1_)/(Ixx + m*L1_^2);
                dx(5) = Fy/m - g - k*zhat - c*dy_;
                dx(6) = -dx(4,1)/L1_;
                

                
            else
                % treat as "falling" mass
                dx(4) = Fx/m;
                dx(5) = Fy/m - g;
                dx(6) = Mz/I;
                dx(1) = dx_;
                dx(2) = dy_;
                dx(3) = dt_;
            end
            
        end
        
    end
    
end