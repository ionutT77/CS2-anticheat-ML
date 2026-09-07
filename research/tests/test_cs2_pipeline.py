import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from anticheat.cs2_data import BASE_COLUMNS, FEATURES, extract_match, time_since, wrap
from anticheat.cs2_models import MatchDetector


def fixture_match(root, future_angle=20):
    n=300;data={c:[] for c in BASE_COLUMNS}
    for tick in range(1,n+1):
        for player in ['A','B']:
            row={c:0. for c in BASE_COLUMNS}
            row.update(tick=tick,steamid=player,is_alive=True,is_warmup_period=False,
                       health=100.,armor_value=100.,active_weapon_ammo=30.,
                       active_weapon_name='AK-47',aim_punch_angle=[0.,0.,0.],
                       team_num=2 if player=='A' else 3)
            if player=='A':
                row['yaw']=179. if tick<=128 else -179.
                if tick>=257:row['yaw']=future_angle
            else:
                row.update(X=-100.,Z=100.)
            for c in BASE_COLUMNS:data[c].append(row[c])
    path=root/'raw'/'cohort'/'sample.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table(data),path)
    events={'player_hurt':[{'tick':257,'attacker_steamid':'A','user_steamid':'B','weapon':'ak47'}],
            'weapon_fire':[{'tick':128,'user_steamid':'A','weapon':'weapon_ak47'},
                           {'tick':257,'user_steamid':'A','weapon':'weapon_ak47'}],
            'cheaters':[{'steamid':'A'}]}
    path.with_suffix('.json').write_text(json.dumps(events))
    return path


def test_geometry_wrap_and_no_post_impact_information(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    p=fixture_match(tmp_path)
    extract_match(str(p))
    out=Path('data/cs2cd/processed/matches/cohort_sample.npz')
    first=np.load(out)['x']
    assert first.shape==(1,256,len(FEATURES))
    assert first[0,128,0]==2 # 179 -> -179 is a two-degree move.
    np.testing.assert_allclose(first[0,:,3],-45,atol=1e-5) # Positive Source pitch looks down.
    assert first[0,:,4].sum()==1 # The impact-tick shot is beyond the input horizon.
    out.unlink();out.with_suffix('.json').unlink()
    fixture_match(tmp_path,future_angle=-140)
    extract_match(str(p))
    np.testing.assert_array_equal(first,np.load(out)['x'])


def test_elapsed_events_are_causal():
    actual=time_since(np.array([0,10,20,30]),[10,100])
    np.testing.assert_allclose(actual,[5,0,10/64,20/64])
    np.testing.assert_allclose(wrap(np.array([359,-359,181,-181])),[-1,1,-179,179])


def test_hierarchical_padding_does_not_change_decision():
    torch.set_num_threads(2);torch.manual_seed(13)
    m=MatchDetector(39,'tcn',True).eval()
    z=torch.randn(1,3,64);mask=torch.ones(1,3,dtype=torch.bool)
    with torch.inference_mode():
        p,_=m.aggregate(z,mask)
        padded=torch.cat([z,torch.randn(1,4,64)*100],dim=1)
        pp,_=m.aggregate(padded,torch.tensor([[True,True,True,False,False,False,False]]))
    torch.testing.assert_close(p,pp,atol=1e-6,rtol=1e-5)
