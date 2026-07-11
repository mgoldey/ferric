from chimerax.core.toolshed import BundleAPI


class _ActiveSiteAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from chimerax.core.commands import register
        from . import cmd

        table = {
            "activesite_charges": (cmd.activesite_charges, cmd.activesite_charges_desc),
            "activesite_embed": (cmd.activesite_embed, cmd.activesite_embed_desc),
            "activesite_energy": (cmd.activesite_energy, cmd.activesite_energy_desc),
            "activesite_properties": (cmd.activesite_properties, cmd.activesite_properties_desc),
            "activesite_bindingenergy": (cmd.activesite_bindingenergy, cmd.activesite_bindingenergy_desc),
        }
        if ci.name not in table:
            raise ValueError(f"unknown command: {ci.name}")
        func, desc = table[ci.name]
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, func)


bundle_api = _ActiveSiteAPI()
