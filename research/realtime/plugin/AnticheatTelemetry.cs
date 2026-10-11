// Telemetry plugin (capture schema v1), CounterStrikeSharp 1.0.376. Runs on the local test
// server; property names checked against the assembly metadata and the server schema.
// Records raw state only; it never scores, kicks or bans.
using System.Text.Json;
using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes.Registration;
using CounterStrikeSharp.API.Modules.Memory;
using CounterStrikeSharp.API.Modules.Utils;
using Microsoft.Extensions.Logging;

namespace AnticheatTelemetry;

public class AnticheatTelemetry : BasePlugin
{
    public override string ModuleName => "AnticheatTelemetry";
    public override string ModuleVersion => "0.2.0-schema1";
    private const int CaptureSchema = 1;
    private const int TickRate = 64;

    private StreamWriter? _out;
    private int _lastTick = -1;
    private int _lastFlushTick;
    private HashSet<ulong> _consent = new();

    public override void Load(bool hotReload)
    {
        // Consent list: one SteamID64 per line. Players not listed are never recorded.
        var consentPath = Path.Combine(ModuleDirectory, "consent.txt");
        if (File.Exists(consentPath))
            _consent = File.ReadAllLines(consentPath)
                .Select(l => l.Trim()).Where(l => ulong.TryParse(l, out _)).Select(ulong.Parse).ToHashSet();
        if (_consent.Count == 0)
            Logger.LogWarning("consent.txt missing or empty: only bots will be recorded.");
        var path = Path.Combine(ModuleDirectory, $"capture_{DateTime.UtcNow:yyyyMMdd_HHmmss}.ndjson");
        _out = new StreamWriter(path) { AutoFlush = false };
        Write(new { type = "header", schema = CaptureSchema, tickrate = TickRate, plugin = ModuleVersion,
                    map = Server.MapName, consent_count = _consent.Count });
        RegisterListener<Listeners.OnTick>(OnTick);
        RegisterListener<Listeners.OnMapEnd>(() => _out?.Flush());
        RegisterEventHandler<EventPlayerHurt>(OnHurt);
        RegisterEventHandler<EventWeaponFire>(OnFire);
        RegisterEventHandler<EventPlayerFootstep>(OnStep);
        RegisterEventHandler<EventPlayerBlind>(OnBlind);
    }

    public override void Unload(bool hotReload)
    {
        _out?.Flush();
        _out?.Dispose();
    }

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
            var punch = TryReadPunch(pawn);
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
                recoil = weapon is null ? 0f : new CCSWeaponBase(weapon.Handle).FlRecoilIndex,
                // null (never 0) when unreadable. punch0/1 = m_predictableBaseAngle (same stand-in
                // as dem_extractor.py); the extra pb_/ub_ fields exist only for the C4 comparison.
                punch0 = punch?.P0, punch1 = punch?.P1,
                pb_vel0 = punch?.V0, pb_vel1 = punch?.V1, pb_tick = punch?.BaseTick,
                ub0 = punch?.U0, ub1 = punch?.U1, ub_tick = punch?.UBaseTick,
                shots = pawn.ShotsFired, clip = weapon?.Clip1 ?? 0,
                weapon = StripPrefix(weapon?.DesignerName),
            });
        }
        // Flush about once a second by tick count (not tick % N, which can be skipped), so a
        // server shutdown loses at most ~1 s and leaves at most one partial final line.
        if (tick - _lastFlushTick >= TickRate) { _out?.Flush(); _lastFlushTick = tick; }
    }

    // The current CS2 build (patch 14190) has no m_aimPunchAngle on the pawn: server.dll and
    // client.dll contain no such string, and reads under that name returned a pointer read as
    // floats. Aim punch now lives in CCSPlayer_AimPunchServices (networked fields confirmed in
    // server.dll): m_predictableBaseAngle/AngleVel/Tick and m_unpredictableBaseAngle/Tick.
    // dem_extractor.py already uses m_predictableBaseAngle as its stand-in for the training
    // feature. Whether it is EQUIVALENT to the per-tick m_aimPunchAngle in CS2CD is unproven
    // (a base angle may not decay per tick), so all components are logged for comparison.
    private const string PunchCls = "CCSPlayer_AimPunchServices";

    private sealed record PunchState(float P0, float P1, float V0, float V1, int BaseTick, float U0, float U1, int UBaseTick);

    private static bool Plausible(QAngle a) =>
        float.IsFinite(a.X) && float.IsFinite(a.Y) && MathF.Abs(a.X) < 180 && MathF.Abs(a.Y) < 180;

    private PunchState? TryReadPunch(CCSPlayerPawn pawn)
    {
        try
        {
            var services = pawn.AimPunchServices;
            if (services is null || services.Handle == IntPtr.Zero) return null;
            var h = services.Handle;
            var angle = Schema.GetDeclaredClass<QAngle>(h, PunchCls, "m_predictableBaseAngle");
            var vel = Schema.GetDeclaredClass<QAngle>(h, PunchCls, "m_predictableBaseAngleVel");
            var unpred = Schema.GetDeclaredClass<QAngle>(h, PunchCls, "m_unpredictableBaseAngle");
            if (!Plausible(angle) || !Plausible(unpred) || !float.IsFinite(vel.X) || !float.IsFinite(vel.Y)) return null;
            return new PunchState(angle.X, angle.Y, vel.X, vel.Y,
                Schema.GetSchemaValue<int>(h, PunchCls, "m_predictableBaseTick"),
                unpred.X, unpred.Y,
                Schema.GetSchemaValue<int>(h, PunchCls, "m_unpredictableBaseTick"));
        }
        catch (Exception e)
        {
            if (!_punchErrorLogged) { _punchErrorLogged = true; Logger.LogWarning("aim punch read failed: {Msg}", e.Message); }
            return null;
        }
    }

    private bool _punchErrorLogged;

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
