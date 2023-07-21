def get_hamiltonian(pb_type, **kwargs):
    if pb_type in ['maxcut']:
        from .maxcut import MaxCut
        return MaxCut(**kwargs)
    elif pb_type in ['vqls']:
        from .vqls import VQLS
        return VQLS(**kwargs)
    elif pb_type in ['vqls_direct']:
        from .vqls_direct import VQLS_direct
        return VQLS_direct(**kwargs)
    else:
        raise "Problem type unspecified!"
