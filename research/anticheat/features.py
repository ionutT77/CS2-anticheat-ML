"""Deterministic, label-free features computed within each record only."""
import numpy as np

WINDOWS = {"full": (0,192), "pre": (0,160), "near": (128,160), "contact": (152,176), "post": (160,192)}


def wrap_degrees(x):
    return (x + 180.0) % 360.0 - 180.0


def engagement_features(x):
    """Input [N,192,5]; return [N,D] and stable names. No fitted quantities."""
    if x.ndim != 3 or x.shape[1:] != (192,5) or not np.isfinite(x).all():
        raise ValueError("Expected finite [N,192,5] float array")
    velocity = x[..., :2]
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:,:1])
    jerk = np.diff(acceleration, axis=1, prepend=acceleration[:,:1])
    error_diff = np.diff(x[...,2:4], axis=1, prepend=x[:,:1,2:4])
    error_diff[...,0] = wrap_degrees(error_diff[...,0])
    speed = np.linalg.norm(velocity, axis=-1)
    aim_error = np.linalg.norm(x[...,2:4], axis=-1)
    signals = np.concatenate([x[...,:4], speed[...,None],
                              np.linalg.norm(acceleration,axis=-1)[...,None],
                              np.linalg.norm(jerk,axis=-1)[...,None], aim_error[...,None],
                              np.linalg.norm(error_diff,axis=-1)[...,None]], axis=-1)
    channels = ["delta_yaw","delta_pitch","error_yaw","error_pitch","speed","acceleration","jerk","aim_error","error_speed"]
    values, names = [], []
    def add(v, n):
        values.append(v[:,None] if v.ndim == 1 else v)
        names.extend([n] if isinstance(n,str) else n)
    for window,(lo,hi) in WINDOWS.items():
        s = signals[:,lo:hi]
        for stat,v in [("mean",s.mean(1)),("std",s.std(1)),("absmean",np.abs(s).mean(1)),
                       ("absq90",np.quantile(np.abs(s),.9,axis=1)),("absmax",np.abs(s).max(1))]:
            add(v,[f"{window}.{c}.{stat}" for c in channels])
        fire = x[:,lo:hi,4]
        a = aim_error[:,lo:hi]
        sp = speed[:,lo:hi]
        add(fire.mean(1), f"{window}.firing.rate")
        add((sp < 1e-5).mean(1), f"{window}.speed.idle_fraction")
        add(np.all(x[:,lo:hi] == 0, axis=-1).mean(1), f"{window}.allzero.fraction")
        for threshold in [.1,.5,1.,2.,5.]:
            add((a < threshold).mean(1), f"{window}.aim_error.below_{threshold}")
        for c,v in [("aim_error",a),("speed",sp),("pitch",x[:,lo:hi,1]),
                    ("acceleration",signals[:,lo:hi,5]),("jerk",signals[:,lo:hi,6])]:
            add((v*fire).sum(1)/np.maximum(fire.sum(1),1), f"{window}.{c}.firing_mean")
        sign = np.sign(x[:,lo:hi,0])
        add(((sign[:,1:]*sign[:,:-1]) < 0).mean(1), f"{window}.delta_yaw.reversal_rate")
        # Stable lag-one movement autocorrelation, zero for constant motion.
        aa = x[:,lo:hi-1,:2]
        bb = x[:,lo+1:hi,:2]
        aa = aa-aa.mean(1,keepdims=True)
        bb = bb-bb.mean(1,keepdims=True)
        corr=(aa*bb).mean(1)/np.maximum(aa.std(1)*bb.std(1),1e-6)
        add(corr,[f"{window}.delta_yaw.autocorr",f"{window}.delta_pitch.autocorr"])
        motion=x[:,lo:hi,:2]
        for lag in [2,3,4,8]:
            aa=motion[:,:-lag];bb=motion[:,lag:]
            aa=aa-aa.mean(1,keepdims=True);bb=bb-bb.mean(1,keepdims=True)
            corr=(aa*bb).mean(1)/np.maximum(aa.std(1)*bb.std(1),1e-6)
            add(corr,[f"{window}.delta_yaw.autocorr_lag{lag}",f"{window}.delta_pitch.autocorr_lag{lag}"])
        spectrum=np.abs(np.fft.rfft(motion-motion.mean(1,keepdims=True),axis=1))[:,1:]**2
        power=spectrum/np.maximum(spectrum.sum(1,keepdims=True),1e-12)
        freq=np.fft.rfftfreq(hi-lo)[1:]
        for stat,v in [("spectral_entropy",-(power*np.log(power+1e-12)).sum(1)/np.log(len(freq))),
                       ("spectral_peak",power.max(1)),("spectral_centroid",(power*freq[None,:,None]).sum(1)),
                       ("spectral_high",power[:,freq>=.25].sum(1)),("spectral_low",power[:,freq<=.125].sum(1))]:
            add(v,[f"{window}.delta_yaw.{stat}",f"{window}.delta_pitch.{stat}"])
        for lag in [1,4]:
            aa=x[:,lo:hi-lag,2:4];bb=motion[:,lag:]
            aa=aa-aa.mean(1,keepdims=True);bb=bb-bb.mean(1,keepdims=True)
            corr=(aa*bb).mean(1)/np.maximum(aa.std(1)*bb.std(1),1e-6)
            add(corr,[f"{window}.tracking_yaw.corr_lag{lag}",f"{window}.tracking_pitch.corr_lag{lag}"])
        for threshold in [.5,1.,2.]:
            add(((a<threshold)*fire).sum(1)/np.maximum(fire.sum(1),1),f"{window}.aim_error.firing_below_{threshold}")
        vv0=motion[:,:-1];vv1=motion[:,1:]
        cosine=(vv0*vv1).sum(-1)/np.maximum(np.linalg.norm(vv0,axis=-1)*np.linalg.norm(vv1,axis=-1),1e-6)
        add(cosine.mean(1),f"{window}.direction.cosine_mean")
        add(cosine.std(1),f"{window}.direction.cosine_std")
        for angle in [2.,5.,10.]:
            snap=(sp[:,:-2]>angle)&(a[:,2:]<1.)
            add(snap.mean(1),f"{window}.snap.speed{angle}_settle1")
    return np.nan_to_num(np.concatenate(values,1),nan=0,posinf=0,neginf=0).astype(np.float32), names


def aggregate_features(engagements, names):
    """Permutation-invariant distribution summaries over a player's engagements."""
    stats = [("mean",engagements.mean(1)),("std",engagements.std(1)),
             ("q10",np.quantile(engagements,.1,axis=1)),("median",np.median(engagements,axis=1)),
             ("q90",np.quantile(engagements,.9,axis=1)),("max",engagements.max(1))]
    return np.concatenate([v for _,v in stats],1).astype(np.float32), [f"{n}__bag_{stat}" for stat,_ in stats for n in names]


def player_features(x):
    e,names = engagement_features(x.reshape(-1,192,5))
    return aggregate_features(e.reshape(len(x),x.shape[1],-1),names)
