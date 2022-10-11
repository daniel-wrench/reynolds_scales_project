# Parameters for processing data

timestamp=      "Epoch"
int_size=       "12H"

omni_path=      "omni/omni_cdaweb/hro2_1min/"
vsw=            "flow_speed"
p=              "Pressure"
Bomni=          "F"
omni_thresh = {
                'flow_speed': [0, 1000],
                'Pressure': [0, 200],
                'F': [0, 50]
            }

electron_path=  "wind/3dp/3dp_elm2/"
ne=             "DENSITY"
Te=             "AVGTEMP"
electron_thresh=None

proton_path=    "wind/3dp/3dp_plsp/"
ni=             "MOM.P.DENSITY"
Ti=             "MOM.P.AVGTEMP"
proton_thresh=  None

mag_path=       "wind/mfi/mfi_h2/"
Bwind=          "BF1"
Bwind_vec=      "BGSE"
Bx=             "BGSE_0"
By=             "BGSE_1"
Bz=             "BGSE_2"
mag_thresh=     None

dt_lr=          "5S"
nlags_lr=       2000
dt_hr=          "0.092S"
nlags_hr=       100
f_min_inertial= 0.01
f_max_inertial= 0.2
f_min_kinetic=  0.5
f_max_kinetic=  2
tau_min=        10
tau_max=        50