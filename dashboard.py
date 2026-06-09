import pygame, sys, os, random, datetime
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── SUMO ──
tools = os.path.join(os.environ.get("SUMO_HOME","C:/Program Files (x86)/Eclipse/Sumo"),"tools")
sys.path.append(tools)
import traci

# ── COLORS ──
BG        = (10, 15, 30)
CARD      = (20, 30, 55)
ROAD      = (38, 38, 38)
MARK      = (255, 215, 0)
WHITE     = (255, 255, 255)
LGRAY     = (150, 150, 150)
DGRAY     = (50,  50,  50)
GREEN     = (0,   200, 90)
DGREEN    = (0,   80,  35)
RED       = (210, 40,  40)
DRED      = (90,  15,  15)
YELLOW    = (255, 200, 0)
DYELLOW   = (90,  70,  0)
ORANGE    = (255, 130, 0)
TEAL      = (0,   180, 180)
BLUE      = (50,  130, 240)
PINK      = (240, 60,  160)
CAR_NORM  = (255, 210, 50)   # normal car
CAR_EMG   = (255, 40,  40)   # emergency car
CAR_PEAK  = (255, 120, 0)    # peak hour car
BLACK     = (0,   0,   0)

W, H = 1280, 720
EPISODES = 10
PEAK_HOURS = [(8,10),(17,20)]

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("DD-DQN Intelligent Traffic Signal Control")
clock = pygame.time.Clock()

FB = pygame.font.SysFont("consolas", 22, bold=True)
FM = pygame.font.SysFont("consolas", 14)
FS = pygame.font.SysFont("consolas", 12)
FT = pygame.font.SysFont("consolas", 17, bold=True)
FL = pygame.font.SysFont("consolas", 28, bold=True)

def is_peak():
    h = datetime.datetime.now().hour
    for s,e in PEAK_HOURS:
        if s<=h<e: return True
    return False

def green_secs(q, peak):
    base = 10 + min(q*2, 30)
    return int(base*1.5) if peak else base

# ── FUZZY ──
def build_fuzzy():
    qi = ctrl.Antecedent(np.arange(0,31,1),'queue')
    di = ctrl.Antecedent(np.arange(0,201,1),'delay')
    ro = ctrl.Consequent(np.arange(-1,1.1,.1),'reward')
    for v,r in [('low',[0,0,10]),('medium',[5,15,25]),('high',[20,30,30])]:
        qi[v]=fuzz.trimf(qi.universe,r)
    for v,r in [('low',[0,0,60]),('medium',[40,100,160]),('high',[140,200,200])]:
        di[v]=fuzz.trimf(di.universe,r)
    for v,r in [('very_low',[-1,-1,-0.5]),('low',[-0.8,-0.4,0]),
                ('medium',[-0.2,0,0.2]),('high',[0,0.4,0.8]),('very_high',[0.5,1,1])]:
        ro[v]=fuzz.trimf(ro.universe,r)
    rules=[
        ctrl.Rule(qi['high']  &di['high'],  ro['very_low']),
        ctrl.Rule(qi['high']  &di['medium'],ro['low']),
        ctrl.Rule(qi['medium']&di['high'],  ro['low']),
        ctrl.Rule(qi['medium']&di['medium'],ro['medium']),
        ctrl.Rule(qi['low']   &di['low'],   ro['very_high']),
        ctrl.Rule(qi['low']   &di['medium'],ro['high']),
        ctrl.Rule(qi['medium']&di['low'],   ro['high']),
        ctrl.Rule(qi['high']  &di['low'],   ro['medium']),
        ctrl.Rule(qi['low']   &di['high'],  ro['low']),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

fz = build_fuzzy()
def fz_reward(q,d):
    try:
        fz.input['queue']=min(float(q),30)
        fz.input['delay']=min(float(d),200)
        fz.compute()
        return float(fz.output['reward'])
    except: return 0.0

# ── DD-DQN ──
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(8,256); self.fc2=nn.Linear(256,128)
        self.vs=nn.Linear(128,1);  self.adv=nn.Linear(128,3)
    def forward(self,x):
        x=torch.relu(self.fc1(x)); x=torch.relu(self.fc2(x))
        v=self.vs(x); a=self.adv(x)
        return v+(a-a.mean(dim=1,keepdim=True))

class Agent:
    def __init__(self):
        self.mem=deque(maxlen=10000); self.gamma=0.95
        self.eps=1.0; self.eps_min=0.01; self.eps_dec=0.99
        self.bs=32
        self.on=Net(); self.tg=Net()
        self.tg.load_state_dict(self.on.state_dict())
        self.opt=optim.Adam(self.on.parameters(),lr=0.0001)
    def act(self,s):
        if random.random()<self.eps: return random.randrange(3)
        with torch.no_grad():
            return self.on(torch.FloatTensor(s).unsqueeze(0)).argmax().item()
    def remember(self,s,a,r,s2,d): self.mem.append((s,a,r,s2,d))
    def replay(self):
        if len(self.mem)<self.bs: return
        b=random.sample(self.mem,self.bs)
        s,a,r,s2,d=zip(*b)
        s=torch.FloatTensor(np.array(s)); a=torch.LongTensor(a)
        r=torch.FloatTensor(r); s2=torch.FloatTensor(np.array(s2))
        d=torch.FloatTensor(d)
        cq=self.on(s).gather(1,a.unsqueeze(1)).squeeze(1)
        nq=self.tg(s2).gather(1,self.on(s2).argmax(1).unsqueeze(1)).squeeze(1)
        tg=r+self.gamma*nq*(1-d)
        loss=nn.MSELoss()(cq,tg.detach())
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.eps>self.eps_min: self.eps*=self.eps_dec
    def update_tg(self): self.tg.load_state_dict(self.on.state_dict())

# ── SUMO ──
CFG="intersection.sumocfg"
PHASES=["GGGggrrrrrGGGggrrrrr","rrrrrGGGggrrrrrGGGgg","GGGggrrrrrrrrrrrrrrr"]
LANES=["north_in_0","north_in_1","south_in_0","south_in_1",
       "east_in_0","east_in_1","west_in_0","west_in_1"]

def get_state():
    return np.array([traci.lane.getLastStepHaltingNumber(l) for l in LANES],dtype=np.float32)
def get_metrics():
    q=sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES)
    d=sum(traci.lane.getWaitingTime(l) for l in LANES)
    t=sum(traci.lane.getLastStepVehicleNumber(l) for l in LANES)
    return q,d,t

# ════════════════════════════════════════
# CAR CLASS — smooth slow movement
# ════════════════════════════════════════
class Car:
    def __init__(self, direction, index, car_type="normal"):
        self.direction = direction
        self.index     = index
        self.type      = car_type
        # Start position far from intersection
        cx,cy,rw = 310,400,60
        spacing = 30
        if direction=="N":
            self.x = cx-rw+12
            self.y = cy-rw-40-index*spacing
            self.dx,self.dy = 0, 1.2    # moves down
        elif direction=="S":
            self.x = cx+14
            self.y = cy+rw+30+index*spacing
            self.dx,self.dy = 0,-1.2    # moves up
        elif direction=="E":
            self.x = cx+rw+30+index*spacing
            self.y = cy-rw+12
            self.dx,self.dy = -1.2, 0   # moves left
        else:  # W
            self.x = cx-rw-50-index*spacing
            self.y = cy+14
            self.dx,self.dy = 1.2, 0    # moves right
        self.moving  = False
        self.waiting = False

    def update(self, signal):
        if signal == "GREEN":
            self.moving  = True
            self.waiting = False
            self.x += self.dx
            self.y += self.dy
        elif signal == "YELLOW":
            # slow down
            self.moving  = True
            self.waiting = False
            self.x += self.dx * 0.4
            self.y += self.dy * 0.4
        else:  # RED
            self.moving  = False
            self.waiting = True

    def draw(self):
        col = CAR_EMG if self.type=="emergency" else \
              CAR_PEAK if self.type=="peak"      else CAR_NORM
        # car body
        pygame.draw.rect(screen, col,    (int(self.x), int(self.y), 22, 14))
        pygame.draw.rect(screen, ORANGE, (int(self.x), int(self.y), 22, 14), 2)
        # windshield
        pygame.draw.rect(screen, BLUE,   (int(self.x)+4, int(self.y)+2, 10, 5))
        # brake light if waiting
        if self.waiting:
            pygame.draw.rect(screen, RED, (int(self.x), int(self.y)+10, 5, 4))

    def is_off_screen(self):
        return (self.x < -50 or self.x > W+50 or
                self.y < -50 or self.y > H+50)

# ════════════════════════════════════════
# TRAFFIC SIGNAL BOX DRAW
# ════════════════════════════════════════
def draw_signal_box(x, y, state, label):
    # Box
    pygame.draw.rect(screen, (20,20,20), (x,y,22,64), border_radius=4)
    # Red
    rc = RED    if state=="RED"    else (50,10,10)
    pygame.draw.circle(screen, rc,     (x+11,y+11), 9)
    # Yellow
    yc = YELLOW if state=="YELLOW" else (50,45,10)
    pygame.draw.circle(screen, yc,     (x+11,y+32), 9)
    # Green
    gc = GREEN  if state=="GREEN"  else (10,50,20)
    pygame.draw.circle(screen, gc,     (x+11,y+53), 9)
    # Label
    screen.blit(FS.render(label,True,WHITE),(x+26,y+22))

# ════════════════════════════════════════
# DRAW 4-WAY ROAD
# ════════════════════════════════════════
def draw_road_only():
    cx,cy,rw = 310,400,60
    # Roads
    pygame.draw.rect(screen,ROAD,(cx-rw,60,rw*2,H-80))
    pygame.draw.rect(screen,ROAD,(60,cy-rw,510,rw*2))
    # Lane markings
    for y in range(110,H-70,20):
        pygame.draw.rect(screen,MARK,(cx-2,y,4,10))
    for x in range(90,540,20):
        pygame.draw.rect(screen,MARK,(x,cy-2,10,4))
    # Intersection box
    pygame.draw.rect(screen,(50,50,50),(cx-rw,cy-rw,rw*2,rw*2))
    # Direction labels
    screen.blit(FM.render("NORTH",True,WHITE),(cx-25,65))
    screen.blit(FM.render("SOUTH",True,WHITE),(cx-25,H-52))
    screen.blit(FM.render("EAST", True,WHITE),(cx+rw+50,cy-rw-35))
    screen.blit(FM.render("WEST", True,WHITE),(65,cy-rw-35))

def draw_signals_on_road(sigs):
    cx,cy,rw = 310,400,60
    draw_signal_box(cx-rw-38, cy-rw-70, sigs["N"], "N")
    draw_signal_box(cx+rw+14, cy+rw+10, sigs["S"], "S")
    draw_signal_box(cx+rw+14, cy-rw-70, sigs["E"], "E")
    draw_signal_box(cx-rw-38, cy+rw+10, sigs["W"], "W")

# ════════════════════════════════════════
# RIGHT PANEL
# ════════════════════════════════════════
def draw_right_panel(ep, counts, sigs, secs,
                     trad_sigs, trad_secs,
                     reason, reason_col,
                     peak, emergency, mode,
                     eps, reward):

    px = 640
    pygame.draw.rect(screen,CARD,(px,58,W-px-5,H-65),border_radius=10)

    # ── EPISODE + MODE ──
    mc = {
        "IDLE":   LGRAY,
        "ANALYSING": YELLOW,
        "RUNNING":   GREEN,
        "DONE":      TEAL,
        "FINISHED":  ORANGE
    }.get(mode, WHITE)
    pygame.draw.rect(screen,mc,(px,58,W-px-5,34),border_radius=10)
    screen.blit(FT.render(f"EPISODE {ep}/{EPISODES}  |  {mode}",True,BLACK),(px+8,64))
    pygame.draw.rect(screen,TEAL,(px,92,W-px-5,2))

    # ── BADGES ──
    pk_c = ORANGE if peak      else DGRAY
    em_c = CAR_EMG if emergency else DGRAY
    pygame.draw.rect(screen,pk_c, (px+2,  97,150,24),border_radius=5)
    pygame.draw.rect(screen,em_c, (px+158,97,150,24),border_radius=5)
    screen.blit(FS.render("PEAK HOUR ON"   if peak      else "OFF PEAK",     True,BLACK),(px+8,  103))
    screen.blit(FS.render("EMERGENCY ON!"  if emergency else "NO EMERGENCY", True,WHITE),(px+164,103))
    screen.blit(FS.render(f"Eps:{eps:.2f}  Reward:{reward:.0f}",True,LGRAY),(px+320,103))
    pygame.draw.rect(screen,TEAL,(px,124,W-px-5,2))

    # ── VEHICLE COUNT ──
    screen.blit(FT.render("VEHICLE COUNT PER LANE",True,YELLOW),(px+5,129))

    dirs = [("N","NORTH"),("S","SOUTH"),("E","EAST"),("W","WEST")]
    for i,(d,name) in enumerate(dirs):
        y   = 152+i*48
        q   = counts.get(d,0)
        sig = sigs.get(d,"RED")
        sc  = GREEN if sig=="GREEN" else YELLOW if sig=="YELLOW" else RED

        pygame.draw.rect(screen,(25,38,65),(px+2,y,W-px-10,42),border_radius=7)
        pygame.draw.rect(screen,sc,        (px+2,y,W-px-10,42),2,border_radius=7)

        # Name + count
        screen.blit(FM.render(f"{name}",True,WHITE),(px+8,y+5))
        screen.blit(FL.render(f"{q:2d}",True,sc),   (px+95,y+4))
        screen.blit(FS.render("cars",True,LGRAY),    (px+150,y+26))

        # Signal pill
        pygame.draw.rect(screen,sc,(px+190,y+10,80,20),border_radius=5)
        screen.blit(FS.render(sig,True,BLACK),(px+194,y+14))

        # Seconds
        sec = secs.get(d,0)
        screen.blit(FM.render(f"{sec:2d}s",True,sc),(px+280,y+12))

        # Bar
        bw = min(q*11,240)
        pygame.draw.rect(screen,(35,35,35),(px+320,y+12,240,18),border_radius=4)
        if bw>0:
            pygame.draw.rect(screen,sc,(px+320,y+12,bw,18),border_radius=4)

        # Priority
        ns_q=counts.get("N",0)+counts.get("S",0)
        ew_q=counts.get("E",0)+counts.get("W",0)
        heavy=(d in["N","S"] and ns_q>=ew_q) or (d in["E","W"] and ew_q>ns_q)
        if heavy and q>0:
            pygame.draw.rect(screen,YELLOW,(px+570,y+10,75,22),border_radius=4)
            screen.blit(FS.render("PRIORITY",True,BLACK),(px+573,y+14))

    pygame.draw.rect(screen,TEAL,(px,345,W-px-5,2))

    # ── AI DECISION ──
    screen.blit(FT.render("AI DECISION (DD-DQN + FUZZY)",True,TEAL),(px+5,350))
    pygame.draw.rect(screen,(10,35,10),(px+2,372,W-px-10,44),border_radius=8)
    pygame.draw.rect(screen,reason_col,(px+2,372,W-px-10,44),2,border_radius=8)
    screen.blit(FM.render(f">> {reason}",True,reason_col),(px+8,378))
    pygame.draw.rect(screen,TEAL,(px,420,W-px-5,2))

    # ── SIGNAL TABLE: AI vs TRADITIONAL ──
    screen.blit(FT.render("SIGNAL SECONDS  (AI  vs  TRADITIONAL)",True,YELLOW),(px+5,426))

    # Headers
    screen.blit(FS.render("LANE",  True,LGRAY),(px+5,  448))
    screen.blit(FS.render("SIGNAL",True,LGRAY),(px+75, 448))
    screen.blit(FS.render("AI SEC",True,GREEN),(px+155,448))
    screen.blit(FS.render("TRAD SEC",True,RED),(px+235,448))
    screen.blit(FS.render("DIFF",  True,ORANGE),(px+325,448))
    screen.blit(FS.render("REASON",True,LGRAY),(px+385,448))
    pygame.draw.rect(screen,TEAL,(px+2,462,W-px-10,1))

    for i,(d,name) in enumerate(dirs):
        y       = 467+i*44
        ai_sec  = secs.get(d,0)
        tr_sec  = trad_secs.get(d,0)
        diff    = ai_sec - tr_sec
        sig     = sigs.get(d,"RED")
        sc      = GREEN if sig=="GREEN" else YELLOW if sig=="YELLOW" else RED
        diff_c  = GREEN if diff>0 else RED if diff<0 else LGRAY
        diff_s  = f"+{diff}s" if diff>0 else f"{diff}s"

        pygame.draw.rect(screen,(22,32,55),(px+2,y,W-px-10,38),border_radius=6)
        pygame.draw.rect(screen,sc,        (px+2,y,W-px-10,38),1,border_radius=6)

        screen.blit(FM.render(name[:5],  True,WHITE),  (px+5,  y+10))
        # Signal dot
        pygame.draw.circle(screen,sc,(px+93,y+18),7)
        screen.blit(FS.render(sig[:1],True,sc),(px+103,y+12))
        screen.blit(FM.render(f"{ai_sec}s", True,GREEN), (px+155,y+10))
        screen.blit(FM.render(f"{tr_sec}s", True,RED),   (px+235,y+10))
        screen.blit(FM.render(diff_s,       True,diff_c),(px+325,y+10))

        # Reason text
        q   = counts.get(d,0)
        ns_q= counts.get("N",0)+counts.get("S",0)
        ew_q= counts.get("E",0)+counts.get("W",0)
        if diff>0:   rsn=f"+{diff}s: heavy lane"
        elif diff<0: rsn=f"{diff}s: light lane"
        else:        rsn="normal"
        if emergency and sig=="GREEN": rsn="EMERGENCY CLEAR"
        if peak and diff>0:            rsn=f"+{diff}s: peak hour"
        screen.blit(FS.render(rsn,True,diff_c),(px+385,y+12))

    pygame.draw.rect(screen,TEAL,(px,643,W-px-5,2))

    # ── TRADITIONAL STATUS ──
    ns_q=counts.get("N",0)+counts.get("S",0)
    ew_q=counts.get("E",0)+counts.get("W",0)
    heavy_side="N-S" if ns_q>=ew_q else "E-W"
    trad_problems=[]
    if emergency:   trad_problems.append("No emergency priority!")
    if peak:        trad_problems.append("No peak hour boost!")
    if ns_q!=ew_q:  trad_problems.append(f"{heavy_side} heavy — ignored!")
    trad_txt = "  |  ".join(trad_problems) if trad_problems else "Running normally"
    pygame.draw.rect(screen,DRED,(px+2,648,W-px-10,30),border_radius=5)
    screen.blit(FS.render(f"TRADITIONAL: {trad_txt}",True,RED),(px+6,655))

    screen.blit(FS.render("Uses: DD-DQN + Fuzzy Reward + SUMO + TraCI",True,LGRAY),(px+5,685))

# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
def main():
    agent   = Agent()
    episode = 1
    mode    = "IDLE"

    # Road center
    cx,cy,rw = 310,400,60

    # Buttons
    start_btn = pygame.Rect(cx-165,H-46,145,36)
    stop_btn  = pygame.Rect(cx+20, H-46,145,36)
    can_start = True
    can_stop  = False

    # Signals
    sigs      = {"N":"RED","S":"RED","E":"RED","W":"RED"}
    secs      = {"N":0,"S":0,"E":0,"W":0}
    trad_sigs = {"N":"GREEN","S":"GREEN","E":"RED","W":"RED"}
    trad_secs = {"N":15,"S":15,"E":15,"W":15}
    counts    = {"N":5,"S":4,"E":6,"W":3}
    reason    = "Press START to begin"
    reason_col= TEAL
    peak      = False
    emergency = False
    action    = 0
    reward    = 0.0
    eps_val   = 1.0

    # Cars list
    cars = []
    def spawn_idle_cars():
        nonlocal cars
        cars=[]
        for i in range(4):
            cars.append(Car("N",i,"normal"))
            cars.append(Car("S",i,"normal"))
            cars.append(Car("E",i,"normal"))
            cars.append(Car("W",i,"normal"))

    def spawn_episode_cars(counts, peak, emergency):
        nonlocal cars
        cars=[]
        for d,name in [("N","NORTH"),("S","SOUTH"),("E","EAST"),("W","WEST")]:
            q = counts.get(d,0)
            for i in range(min(q,8)):
                if emergency and i==0:
                    t="emergency"
                elif peak and i<2:
                    t="peak"
                else:
                    t="normal"
                cars.append(Car(d,i,t))

    spawn_idle_cars()

    # SUMO
    traci_open   = False
    sim_state    = np.zeros(8)
    total_reward = 0.0
    step_count   = 0

    # Timers
    analyse_timer = 0
    run_timer     = 0
    run_duration  = 80
    trad_timer    = 0
    trad_phase    = 0

    running = True
    while running:
        dt = clock.tick(15)

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                if traci_open:
                    try: traci.close()
                    except: pass
                pygame.quit(); sys.exit()

            if ev.type==pygame.MOUSEBUTTONDOWN:
                # START
                if start_btn.collidepoint(ev.pos) and can_start:
                    if mode in ["IDLE","DONE"]:
                        # New counts
                        counts = {
                            "N":random.randint(3,16),
                            "S":random.randint(3,16),
                            "E":random.randint(3,16),
                            "W":random.randint(3,16)
                        }
                        peak      = is_peak()
                        emergency = random.randint(1,6)==1
                        reason    = "DD-DQN analysing lane vehicle counts..."
                        reason_col= YELLOW

                        # Spawn cars — all waiting
                        spawn_episode_cars(counts,peak,emergency)
                        sigs = {"N":"RED","S":"RED","E":"RED","W":"RED"}
                        secs = {"N":0,"S":0,"E":0,"W":0}

                        # Start SUMO
                        if traci_open:
                            try: traci.close()
                            except: pass
                        traci.start(["sumo","-c",CFG,"--no-warnings","true"])
                        traci_open   = True
                        sim_state    = get_state()
                        total_reward = 0.0
                        step_count   = 0
                        analyse_timer= 0
                        mode         = "ANALYSING"
                        can_start    = False
                        can_stop     = True

                # STOP
                if stop_btn.collidepoint(ev.pos) and can_stop:
                    if traci_open:
                        try: traci.close()
                        except: pass
                        traci_open=False
                    mode      ="DONE"
                    can_start =True
                    can_stop  =False
                    reason    =f"Stopped. Press START for Episode {min(episode+1,EPISODES)}"
                    reason_col=TEAL
                    sigs={"N":"RED","S":"RED","E":"RED","W":"RED"}

        # ════ STATE MACHINE ════

        if mode=="IDLE":
            # Idle: few cars slowly drifting to signal line
            trad_timer+=1
            if trad_timer>30:
                trad_timer=0
                trad_phase=(trad_phase+1)%2
            if trad_phase==0:
                trad_sigs={"N":"GREEN","S":"GREEN","E":"RED","W":"RED"}
            else:
                trad_sigs={"N":"RED","S":"RED","E":"GREEN","W":"GREEN"}
            sigs=trad_sigs.copy()
            trad_secs={"N":15,"S":15,"E":15,"W":15}
            secs=trad_secs.copy()
            for c in cars:
                c.update(sigs.get(c.direction,"RED"))
            # Respawn if off screen
            cars=[c for c in cars if not c.is_off_screen()]
            if len(cars)<12:
                d=random.choice(["N","S","E","W"])
                cars.append(Car(d,0,random.choice(["normal","normal","normal","peak"])))
            reason="Press START to begin episode"
            reason_col=LGRAY

        elif mode=="ANALYSING":
            # All cars waiting at RED
            analyse_timer+=1
            sigs={"N":"RED","S":"RED","E":"RED","W":"RED"}
            for c in cars: c.update("RED")

            # AI decision after 2 seconds
            if analyse_timer>40:
                ns_q=counts["N"]+counts["S"]
                ew_q=counts["E"]+counts["W"]

                if emergency:
                    action=2
                    reason="EMERGENCY VEHICLE! All lanes CLEAR — priority given"
                    reason_col=CAR_EMG
                elif ns_q<=3 and ew_q>6:
                    action=0
                    reason=f"SIDE LANE PRIORITY: N-S only {ns_q} cars — clearing side first"
                    reason_col=PINK
                elif ew_q<=3 and ns_q>6:
                    action=1
                    reason=f"SIDE LANE PRIORITY: E-W only {ew_q} cars — clearing side first"
                    reason_col=PINK
                elif peak:
                    if ns_q>=ew_q:
                        action=0
                        reason=f"PEAK HOUR: N-S has {ns_q} cars — extra green time given"
                    else:
                        action=1
                        reason=f"PEAK HOUR: E-W has {ew_q} cars — extra green time given"
                    reason_col=ORANGE
                else:
                    action=agent.act(sim_state)
                    names=["N-S GREEN","E-W GREEN","ALL GREEN"]
                    reason=f"DD-DQN: {names[action]} — heavy side gets more seconds"
                    reason_col=GREEN

                # Compute seconds
                ns_s = green_secs(ns_q,peak)
                ew_s = green_secs(ew_q,peak)
                trad_s = 15  # traditional fixed

                if action==0:
                    sigs={"N":"GREEN","S":"GREEN","E":"RED","W":"RED"}
                    secs={"N":ns_s,"S":ns_s,"E":ew_s,"W":ew_s}
                elif action==1:
                    sigs={"N":"RED","S":"RED","E":"GREEN","W":"GREEN"}
                    secs={"N":ns_s,"S":ns_s,"E":ew_s,"W":ew_s}
                else:
                    sigs={"N":"GREEN","S":"GREEN","E":"GREEN","W":"GREEN"}
                    secs={"N":ns_s,"S":ns_s,"E":ew_s,"W":ew_s}

                trad_sigs={"N":"GREEN","S":"GREEN","E":"RED","W":"RED"}
                trad_secs={"N":trad_s,"S":trad_s,"E":trad_s,"W":trad_s}

                run_timer    = 0
                run_duration = max(ns_s,ew_s)*2
                mode         = "RUNNING"
                eps_val      = agent.eps

        elif mode=="RUNNING":
            run_timer+=1

            # Yellow phase near end
            if run_timer>run_duration-20:
                for d in list(sigs):
                    if sigs[d]=="GREEN":
                        sigs[d]="YELLOW"
                        secs[d]=3

            # Update cars
            for c in cars:
                c.update(sigs.get(c.direction,"RED"))
            cars=[c for c in cars if not c.is_off_screen()]

            # SUMO step
            if traci_open:
                try:
                    traci.trafficlight.setRedYellowGreenState("center",PHASES[action])
                    for _ in range(3): traci.simulationStep()
                    q,d_,t=get_metrics()
                    ns=get_state()
                    r=(-0.4*q - 0.4*d_ + 0.2*t) + fz_reward(q/30,d_/200)*10
                    agent.remember(sim_state,action,r,ns,False)
                    agent.replay()
                    sim_state    =ns
                    total_reward+=r
                    step_count  +=1
                    reward       =total_reward
                    eps_val      =agent.eps
                except: pass

            # Done
            if run_timer>run_duration:
                if traci_open:
                    try: traci.close()
                    except: pass
                    traci_open=False
                if episode%5==0: agent.update_tg()
                print(f"Episode {episode} | Reward:{total_reward:.1f}")
                sigs={"N":"RED","S":"RED","E":"RED","W":"RED"}
                mode     ="DONE"
                can_start=True
                can_stop =False
                reason   =f"Episode {episode} complete! Press START for Episode {min(episode+1,EPISODES)}"
                reason_col=TEAL
                episode  +=1
                if episode>EPISODES:
                    mode="FINISHED"
                    reason="ALL 10 EPISODES COMPLETE!"
                    reason_col=GREEN
                    can_start=False

        elif mode=="DONE":
            for c in cars: c.update("RED")

        elif mode=="FINISHED":
            pass

        # ════ DRAW ════
        screen.fill(BG)

        # Header
        pygame.draw.rect(screen,CARD,(0,0,W,54))
        pygame.draw.rect(screen,TEAL,(0,52,W,2))
        screen.blit(FB.render(
            "DD-DQN INTELLIGENT TRAFFIC SIGNAL CONTROL SYSTEM",
            True,TEAL),(12,14))
        screen.blit(FS.render(
            datetime.datetime.now().strftime("%H:%M:%S"),
            True,LGRAY),(W-80,18))

        # Road
        draw_road_only()
        draw_signals_on_road(sigs)

        # Cars
        for c in cars: c.draw()

        # Right panel
        draw_right_panel(
            episode if episode<=EPISODES else EPISODES,
            counts, sigs, secs,
            trad_sigs, trad_secs,
            reason, reason_col,
            peak, emergency, mode,
            eps_val, reward)

        # Buttons
        for rect,lbl,can,ac,ic in [
            (start_btn,"▶  START",can_start,DGREEN,GREEN),
            (stop_btn, "■  STOP", can_stop, DRED,  RED)
        ]:
            bc=ac if can else DGRAY
            tc=ic if can else LGRAY
            pygame.draw.rect(screen,bc,rect,border_radius=10)
            pygame.draw.rect(screen,tc,rect,2,border_radius=10)
            screen.blit(FT.render(lbl,True,tc),(rect.x+18,rect.y+8))

        # Mode label
        mlbl={
            "IDLE":      "WAITING — Press START",
            "ANALYSING": "AI ANALYSING LANES...",
            "RUNNING":   "AI SIGNAL ACTIVE — VEHICLES MOVING",
            "DONE":      "EPISODE DONE — Press START",
            "FINISHED":  "ALL EPISODES COMPLETE!"
        }.get(mode,"")
        mcol={
            "IDLE":LGRAY,"ANALYSING":YELLOW,
            "RUNNING":GREEN,"DONE":TEAL,"FINISHED":ORANGE
        }.get(mode,WHITE)
        tw=FS.size(mlbl)[0]
        pygame.draw.rect(screen,(15,20,40),(cx-tw//2-8,57,tw+16,22),border_radius=4)
        pygame.draw.rect(screen,mcol,      (cx-tw//2-8,57,tw+16,22),1,border_radius=4)
        screen.blit(FS.render(mlbl,True,mcol),(cx-tw//2,61))

        pygame.display.flip()

if __name__=="__main__":
    main()
