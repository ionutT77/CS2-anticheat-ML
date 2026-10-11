// UNCOMPILED DRAFT (schema v1). Written without the CounterStrikeSharp package restored, so
// every API name below is unverified and must be checked against the installed CSS version
// (see PHASE0_PARITY.md). Records raw state only; it never scores, kicks or bans.
using System.Text.Json;
using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes.Registration;
using CounterStrikeSharp.API.Modules.Utils;

namespace AnticheatTelemetry;

public class AnticheatTelemetry : BasePlugin
{
    public override string ModuleName => "AnticheatTelemetry";
    public override string ModuleVersion => "0.1.0-schema1";
    private const int Schema = 1;
    private const int TickRate = 64;

    private StreamWriter? _out;
    private int _lastTick = -1;
    private HashSet<ulong> _consent = new();

    public override void Load(bool hotReload)
    {
        // Consent list: one SteamID64 per line. Players not listed are never recorded.
        var consentPath = Path.Combine(ModuleDirectory, "consent.txt");
        if (File.Exists(consentPath))
            _consent = File.ReadAllLines(consentPath)
                .Select(l => l.Trim()).Where(l => ulong.TryParse(l, out _)).Select(ulong.Parse).ToHashSet();
        if (_consent.Count == 0)
        {
            Logger.LogError("consent.txt missing or empty: recording disabled.");
            return;
        }
        var path = Path.Combine(ModuleDirectory, $"capture_{DateTime.UtcNow:yyyyMMdd_HHmmss}.ndjson");
        _out = new StreamWriter(path) { AutoFlush = false };
        Write(new { type = "header", schema = Schema, tickrate = TickRate, plugin = ModuleVersion,
                    map = Server.MapName, consent_count = _consent.Count });
        RegisterListener<Listeners.OnTick>(OnTick);
        RegisterEventHandler<EventPlayerHurt>(OnHurt);
        RegisterEventHandler<EventWeaponFire>(OnFire);
        RegisterEventHandler<EventPlayerFootstep>(OnStep);
        RegisterEventHandler<EventPlayerBlind>(OnBlind);
    }

    public override void Unload(bool hotReload) => _out?.Dispose();

    private void Write(object row) => _out?.WriteLine(JsonSerializer.Serialize(row));

    private bool Consented(CCSPlayerController? p) =>
        p is { IsValid: true, IsBot: false } && _consent.Contains(p.SteamID);

    // Bots are recorded by slot for capture testing only; real players need consent.
    private bool Recordable(CCSPlayerController? p) => p is { IsValid: true } && (p.IsBot || Consented(p));

    private void OnTick()
    {
        var tick = Server.TickCount;
        if (tick == _lastTick) return; // OnTick is per frame; keep one sample per game tick
        _lastTick = tick;
        var rules = Utilities.FindAllEntitiesByDesignerName<CCSGameRulesProxy>("cs_gamerules").FirstOrDefault()?.GameRules;
        foreach (var p in Utilities.GetPlayers())
        {
            if (!Recordable(p)) continue;
            var pawn = p.PlayerPawn.Value;
            if (pawn is null || pawn.AbsOrigin is null || pawn.AbsVelocity is null || pawn.EyeAngles is null) continue;
            var weapon = pawn.WeaponServices?.ActiveWeapon.Value;
            var duck = pawn.MovementServices is null ? 0f
                : new CCSPlayer_MovementServices(pawn.MovementServices.Handle).DuckAmount;
            Write(new
            {
                type = "state", tick, slot = p.Slot, steamid = p.IsBot ? 0UL : p.SteamID,
                yaw = pawn.EyeAngles.Y, pitch = pawn.EyeAngles.X,
                x = pawn.AbsOrigin.X, y = pawn.AbsOrigin.Y, z = pawn.AbsOrigin.Z,
                avx = pawn.AbsVelocity.X, avy = pawn.AbsVelocity.Y, avz = pawn.AbsVelocity.Z,
                alive = pawn.LifeState == (byte)LifeState_t.LIFE_ALIVE ? 1 : 0,
                warmup = rules?.WarmupPeriod == true ? 1 : 0, team = p.TeamNum,
                round = rules?.TotalRoundsPlayed ?? -1, scoped = pawn.IsScoped ? 1 : 0, duck,
                flags = pawn.Flags, health = pawn.Health, armor = pawn.ArmorValue,
                recoil = weapon is null ? 0f : new CCSWeaponBase(weapon.Handle).RecoilIndex,
                punch0 = pawn.AimPunchAngle.X, punch1 = pawn.AimPunchAngle.Y,
                shots = pawn.ShotsFired, clip = weapon?.Clip1 ?? 0,
                weapon = StripPrefix(weapon?.DesignerName),
            });
        }
        if (tick % (TickRate * 5) == 0) _out?.Flush();
    }

    private static string StripPrefix(string? name) =>
        name is not null && name.StartsWith("weapon_") ? name["weapon_".Length..] : name ?? "";

    private int SlotOf(CCSPlayerController? p) => p is { IsValid: true } ? p.Slot : -1;

    private HookResult OnHurt(EventPlayerHurt e, GameEventInfo _)
    {
        // Logged at the server tick the event fires; the offset against state samples is
        // measured by check_capture.py (C1), not assumed.
        Write(new { type = "event", name = "player_hurt", tick = Server.TickCount,
                    attacker = SlotOf(e.Attacker), victim = SlotOf(e.Userid),
                    weapon = e.Weapon, dmg_health = e.DmgHealth, health = e.Health });
        return HookResult.Continue;
    }

    private HookResult OnFire(EventWeaponFire e, GameEventInfo _)
    {
        Write(new { type = "event", name = "weapon_fire", tick = Server.TickCount,
                    user = SlotOf(e.Userid), weapon = e.Weapon });
        return HookResult.Continue;
    }

    private HookResult OnStep(EventPlayerFootstep e, GameEventInfo _)
    {
        Write(new { type = "event", name = "player_footstep", tick = Server.TickCount, user = SlotOf(e.Userid) });
        return HookResult.Continue;
    }

    private HookResult OnBlind(EventPlayerBlind e, GameEventInfo _)
    {
        Write(new { type = "event", name = "player_blind", tick = Server.TickCount,
                    user = SlotOf(e.Userid), blind_duration = e.BlindDuration });
        return HookResult.Continue;
    }
}
