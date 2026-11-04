import mne

def get_lead_field_matrix(sfreq):
    montage = mne.channels.make_standard_montage("colin27_1020")
    chs = [c for c in montage.ch_names if c not in ["T3", "T4", "T5", "T6"]]
    info = mne.create_info(chs, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    src = mne.setup_volume_source_space(
        sphere=sphere, pos=15.0, exclude=30.0, add_interpolator=False
    )
    
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0
    )
    return fwd["sol"]["data"]