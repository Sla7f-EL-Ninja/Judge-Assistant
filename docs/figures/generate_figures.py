import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines
import numpy as np
import os

NAVY='#1a2e4a'; GOLD='#c9a84c'; WHITE='#ffffff'; LGRAY='#f5f5f5'; MGRAY='#cccccc'
RED='#c0392b'; GREEN='#27ae60'; BLUE='#2471a3'; LBLUE='#d6eaf8'
SAVE_DIR = r'D:\FUCK!!\Grad\Code\docs\figures'
os.makedirs(SAVE_DIR, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10})

def save_fig(fig, fname):
    fig.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'Saved: {fname}')

def rbox(ax, x, y, w, h, text, fc=LGRAY, ec=NAVY, tc=NAVY, fs=9, bold=False, lw=1.5, zorder=2, radius=0.05):
    box = FancyBboxPatch((x,y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                         fc=fc, ec=ec, lw=lw, zorder=zorder, clip_on=False)
    ax.add_patch(box)
    if text:
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', zorder=zorder+1,
                multialignment='center')

def arrow(ax, x0,y0,x1,y1, label='', color=NAVY, lw=1.5, fs=8, labelpad=(0.05,0)):
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                annotation_clip=False)
    if label:
        ax.text((x0+x1)/2+labelpad[0], (y0+y1)/2+labelpad[1], label,
                ha='left', va='center', fontsize=fs, color=color)

def diamond(ax, cx, cy, w, h, text, fc=LGRAY, ec=NAVY, tc=NAVY, fs=8):
    hw,hh=w/2,h/2
    ax.add_patch(plt.Polygon([[cx,cy+hh],[cx+hw,cy],[cx,cy-hh],[cx-hw,cy]],
                              fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold', zorder=3, multialignment='center')

def caption(fig, text):
    fig.text(0.5, 0.005, text, ha='center', va='bottom', fontsize=10,
             color=NAVY, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc=LGRAY, ec=MGRAY, alpha=0.9))


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.1
# ─────────────────────────────────────────────────────────────────
def fig3_1():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off')
    ax.set_facecolor(WHITE)

    # Layer bands
    for yb,yt,alpha in [(0.2,2.8,0.04),(3.2,5.8,0.04),(6.2,8.5,0.04)]:
        ax.add_patch(FancyBboxPatch((0.2,yb),13.6,yt-yb,boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=LGRAY,ec=MGRAY,lw=0.8,alpha=0.6,zorder=0))

    # Layer labels
    for y,lbl in [(7.3,'Presentation\nLayer'),(4.5,'AI Orchestration\nLayer'),(1.5,'Persistence\nLayer')]:
        ax.text(0.05,y,lbl,ha='center',va='center',fontsize=8,color=GOLD,fontweight='bold',
                rotation=90,transform=ax.transData)

    # Layer 1
    rbox(ax,1.5,6.8,2.8,1.0,'Client\n(Judge Browser)',fc=GOLD,ec=NAVY,tc=NAVY,fs=10,bold=True)
    rbox(ax,6.0,6.8,3.5,1.0,'Node.js BFF\n(Port 3000)',fc=GOLD,ec=NAVY,tc=NAVY,fs=10,bold=True)
    arrow(ax,4.3,7.3,6.0,7.3,label='HTTP / SSE / Cookie JWT',fs=7,labelpad=(0,0.15))

    # Layer 2
    rbox(ax,0.5,4.2,2.8,1.0,'FastAPI AI Server\n(Port 8000)',fc=NAVY,ec=NAVY,tc=WHITE,fs=9,bold=True)
    rbox(ax,4.0,4.0,3.5,1.4,'Supervisor Graph\n(LangGraph)\n17 nodes / SSE stream',fc=NAVY,ec=GOLD,tc=WHITE,fs=9,bold=True,lw=2.5)
    arrow(ax,3.3,4.7,4.0,4.7)

    # Agents
    agents=['OCR','Civil Law\nRAG','Case Doc\nRAG','Chat\nReasoner','Summarizer']
    for i,ag in enumerate(agents):
        x=8.0+i*1.1
        rbox(ax,x,4.1,1.0,1.2,ag,fc=LBLUE,ec=NAVY,tc=NAVY,fs=7.5)
    arrow(ax,7.5,4.7,8.0,4.7)
    ax.text(8.55,5.45,'Agent Pool',ha='center',va='bottom',fontsize=7,color=NAVY,style='italic')

    # Layer 1→2 arrow
    arrow(ax,7.75,6.8,7.75,5.5,label='',color=NAVY)

    # Layer 2→3 arrow
    arrow(ax,4.5,4.0,4.5,2.8,color=NAVY)

    # Layer 3
    stores=[
        ('MongoDB\n(cases, docs,\nconversations,\nmemory)',0.5),
        ('Qdrant\n(judicial_docs\ncase_docs\n1024-dim COSINE)',4.0),
        ('Redis\n(response cache\nSHA-256 keyed)',7.5),
        ('MinIO\n(binary file\nstorage)',11.0),
    ]
    for txt,x in stores:
        rbox(ax,x,0.5,2.6,2.0,txt,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8)

    caption(fig,'Figure 3.1: End-to-end system architecture of the Hakim AI legal assistant')
    save_fig(fig,'fig3_1_system_architecture.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.2
# ─────────────────────────────────────────────────────────────────
def fig3_2():
    fig, ax = plt.subplots(figsize=(16,10))
    ax.set_xlim(0,16); ax.set_ylim(0,10)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def coll_box(ax, x, y, w, name, fields):
        hdr_h=0.55; row_h=0.32; body_h=len(fields)*row_h+0.1
        rbox(ax,x,y+body_h,w,hdr_h,name,fc=NAVY,ec=NAVY,tc=WHITE,fs=8.5,bold=True)
        rbox(ax,x,y,w,body_h,'',fc=LGRAY,ec=NAVY,tc=NAVY,fs=7.5)
        for i,f in enumerate(fields):
            ax.text(x+0.15,y+body_h-(i+0.5)*row_h,f,ha='left',va='center',fontsize=7,color=NAVY)
        return body_h+hdr_h

    # Top row y=5.5
    coll_box(ax,0.3,5.5,3.2,'cases',
        ['_id (ObjectId)','title (str)','description (str)','status (str)','user_id (str)','created_at (datetime)'])
    coll_box(ax,4.0,5.5,3.2,'files',
        ['_id (ObjectId)','case_id (ObjectId)','filename (str)','mimetype (str)','storage_path (str)',
         'size_bytes (int)','uploaded_at (datetime)'])
    coll_box(ax,7.7,5.5,3.5,'documents',
        ['_id (ObjectId)','case_id (ObjectId)','file_ids[] (ObjectId[])','doc_type (str)',
         'extracted_text (str)','confidence_scores[] (float[])','classified_at (datetime)'])
    coll_box(ax,11.7,5.5,3.5,'conversations',
        ['_id (ObjectId)','case_id (ObjectId)',
         'turns[] (array):','  query (str)','  response (str)','  timestamp (datetime)','created_at (datetime)'])

    # Bottom row y=1.0
    coll_box(ax,0.3,1.2,3.5,'summaries',
        ['_id (ObjectId)','case_id (ObjectId)','rendered_brief (str)','party_manifest{} (dict)','created_at (datetime)'])
    coll_box(ax,4.5,1.2,3.5,'supervisor_checkpoints',
        ['_id (ObjectId)','thread_id (str)','checkpoint_id (str)','state_blob (binary)','created_at (datetime)'])
    coll_box(ax,8.5,1.2,4.0,'supervisor_memory_store',
        ['_id (ObjectId)','namespace (3-tuple)','key (str)','value{} (dict)','updated_at (datetime)'])

    # Relationship arrows
    def rel_arrow(ax,x0,y0,x1,y1,lbl,color=GOLD):
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),
                    arrowprops=dict(arrowstyle='->',color=color,lw=1.2,linestyle='dashed'),
                    annotation_clip=False)
        mx,my=(x0+x1)/2,(y0+y1)/2
        ax.text(mx,my+0.1,lbl,ha='center',va='bottom',fontsize=6.5,color=color,
                bbox=dict(fc=WHITE,ec='none',pad=1))

    rel_arrow(ax,1.9,5.5,5.6,7.85,'1..* files per case')
    rel_arrow(ax,1.9,5.5,9.45,7.85,'1..* docs per case')
    rel_arrow(ax,1.9,5.5,13.45,7.85,'1..* convs per case')
    rel_arrow(ax,1.9,5.5,2.05,3.8,'0..1 summary per case')
    rel_arrow(ax,7.1,7.0,9.5,7.2,'file_ids[] ref',color=BLUE)

    caption(fig,'Figure 3.2: MongoDB collection schema and relationships')
    save_fig(fig,'fig3_2_mongodb_er.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.3
# ─────────────────────────────────────────────────────────────────
def fig3_3():
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlim(0,14); ax.set_ylim(0,10)
    ax.axis('off'); ax.set_facecolor(WHITE)

    actors=['Client','FastAPI','Redis','Supervisor\nGraph','SSE Stream']
    xs=[1.2,3.5,5.8,8.5,11.5]
    colors=[GOLD,NAVY,GREEN,NAVY,BLUE]

    # Actor boxes + lifelines
    for i,(name,x,c) in enumerate(zip(actors,xs,colors)):
        rbox(ax,x-0.7,9.0,1.4,0.7,name,fc=c,ec=NAVY,tc=WHITE,fs=8.5,bold=True)
        ax.plot([x,x],[0.5,9.0],color=NAVY,lw=1,linestyle='dashed',zorder=1)

    def seq_arrow(ax,i0,i1,y,label,dashed=False,color=NAVY):
        x0,x1=xs[i0],xs[i1]
        style = 'dashed' if dashed else 'solid'
        ax.annotate('',xy=(x1,y),xytext=(x0,y),
                    arrowprops=dict(arrowstyle='->',color=color,lw=1.3,linestyle=style),
                    annotation_clip=False)
        mx=(x0+x1)/2
        ax.text(mx,y+0.12,label,ha='center',va='bottom',fontsize=7.5,color=color)

    seq_arrow(ax,0,1,8.2,'1. POST /api/v1/query + JWT Bearer')
    seq_arrow(ax,1,2,7.4,'2. LOOKUP sha256(query+case_id)')

    # Cache HIT box
    ax.add_patch(FancyBboxPatch((3.0,6.2),4.5,0.9,boxstyle='round,pad=0,rounding_size=0.05',
                                 fc='#eafaf1',ec=GREEN,lw=1,linestyle='dashed'))
    ax.text(3.1,7.0,'Cache HIT',ha='left',va='top',fontsize=7,color=GREEN,fontweight='bold')
    seq_arrow(ax,2,1,6.6,'3a. cached SSE payload',dashed=True,color=GREEN)
    seq_arrow(ax,1,0,6.2,'3a. SSE replay (cached:true)',dashed=True,color=GREEN)

    # Cache MISS box
    ax.add_patch(FancyBboxPatch((3.0,2.2),9.5,3.7,boxstyle='round,pad=0,rounding_size=0.05',
                                 fc='#fef9e7',ec=GOLD,lw=1,linestyle='dashed'))
    ax.text(3.1,5.8,'Cache MISS',ha='left',va='top',fontsize=7,color=GOLD,fontweight='bold')

    seq_arrow(ax,1,3,5.4,'4. stream(SupervisorState)')
    seq_arrow(ax,3,4,4.7,'5. progress events (per node) [loop]')
    seq_arrow(ax,4,1,4.0,'6. SSE chunks (streamed)')
    seq_arrow(ax,1,0,3.4,'6. SSE chunks → Client')
    seq_arrow(ax,1,2,2.7,'7. STORE result (TTL 5min)')
    seq_arrow(ax,1,0,2.2,'8. SSE done event')

    caption(fig,'Figure 3.3: Request lifecycle sequence diagram')
    save_fig(fig,'fig3_3_request_lifecycle.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.4
# ─────────────────────────────────────────────────────────────────
def fig3_4():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def screen(ax,x,y,w,h,text,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8.5):
        rbox(ax,x,y,w,h,text,fc=fc,ec=ec,tc=tc,fs=fs,bold=True)

    def guard_badge(ax,x,y):
        rbox(ax,x,y,1.4,0.3,'Session Guard\n(GET /api/auth)',fc=GOLD,ec=NAVY,tc=NAVY,fs=6,radius=0.04)

    # Primary flow (center-left column)
    screen(ax,1.0,7.5,2.5,0.7,'Login',fc=GOLD,ec=NAVY,tc=NAVY)
    screen(ax,1.0,5.8,2.5,0.7,'OTP Verification',fc=GOLD,ec=NAVY,tc=NAVY)
    screen(ax,1.0,4.0,2.5,0.7,'Dashboard')
    guard_badge(ax,1.05,4.72)
    screen(ax,1.0,2.2,2.5,0.7,'Case Chat')
    guard_badge(ax,1.05,2.92)

    arrow(ax,2.25,7.5,2.25,6.5,color=GOLD,lw=2)
    arrow(ax,2.25,5.8,2.25,4.7,color=GOLD,lw=2)
    arrow(ax,2.25,4.0,2.25,2.9,color=GOLD,lw=2)

    # Dashboard branches (right)
    branches=[
        (4.5,7.2,'OCR Viewer'),
        (4.5,5.8,'Legal Search'),
        (4.5,4.4,'Case Summary'),
        (4.5,3.0,'Settings'),
    ]
    for x,y,lbl in branches:
        screen(ax,x,y,2.2,0.65,lbl)
        guard_badge(ax,x+0.05,y+0.68)
        arrow(ax,3.5,4.35,x,y+0.32,color=NAVY,lw=1.3)

    # Forgot password branch (left)
    screen(ax,6.5,7.5,2.2,0.65,'Forgot Password',fc=LGRAY)
    screen(ax,6.5,6.0,2.2,0.65,'OTP Reset',fc=LGRAY)
    screen(ax,6.5,4.5,2.2,0.65,'Reset Password',fc=LGRAY)
    arrow(ax,1.0,7.85,6.5,7.85,color=NAVY,lw=1.2)
    arrow(ax,7.6,7.5,7.6,6.65,color=NAVY,lw=1.2)
    arrow(ax,7.6,6.0,7.6,5.15,color=NAVY,lw=1.2)
    # Loop back arrow
    ax.annotate('',xy=(1.0,7.5),xytext=(6.5,4.8),
                arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.2,
                               connectionstyle='arc3,rad=0.3'),annotation_clip=False)
    ax.text(4.5,5.7,'← loop back',ha='center',fontsize=7,color=NAVY,style='italic')

    ax.set_title('Navigation Flow — Hakim Legal AI',fontsize=13,color=NAVY,fontweight='bold',pad=10)
    caption(fig,'Figure 3.4: Front-end screen navigation flow with session guard')
    save_fig(fig,'fig3_4_navigation_flow.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.5
# ─────────────────────────────────────────────────────────────────
def fig3_5():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    # Nav bar
    rbox(ax,0,8.2,14,0.7,'',fc=NAVY,ec=NAVY)
    ax.text(0.3,8.55,'Hakim | AI Legal Assistant',ha='left',va='center',fontsize=10,color=GOLD,fontweight='bold')
    ax.text(13.7,8.55,'Judge Ahmed Al-Rashidi   2024-01-20',ha='right',va='center',fontsize=8,color=WHITE)
    rbox(ax,12.5,8.3,1.1,0.3,'Logout',fc=RED,ec=RED,tc=WHITE,fs=7.5)

    # Page title
    ax.text(7,7.85,'Case Management /  ' + u'إدارة القضايا',
            ha='center',va='center',fontsize=12,color=GOLD,fontweight='bold')

    # Filter bar
    rbox(ax,0.3,7.2,3.0,0.5,'Status: All  v',fc=WHITE,ec=MGRAY,tc=NAVY,fs=8.5)
    rbox(ax,11.5,7.2,2.2,0.5,'+ New Case',fc=GOLD,ec=GOLD,tc=NAVY,fs=9,bold=True)

    # Table header
    cols=['Case Title','Description','Docs','Status','Date','Actions']
    widths=[2.5,4.0,0.8,1.5,1.8,1.1]
    x=0.3
    for col,w in zip(cols,widths):
        rbox(ax,x,6.6,w,0.55,col,fc=NAVY,ec=NAVY,tc=WHITE,fs=8,bold=True)
        x+=w+0.05

    # Data rows
    rows=[
        ('١٢٣/٢٠٢٤/مدني','Real estate dispute...','3','Open','2024-01-15',True,True),
        ('٤٥٦/٢٠٢٤/تجاري','Compensation claim...','5','[dropdown]','2024-02-01',False,False),
        ('789/2024/civil','Appeal case filing...','2','Closed','2024-02-10',False,False),
        ('012/2024/admin','Administrative review','1','Archived','2024-03-01',False,False),
    ]
    for ri,row in enumerate(rows):
        bg=WHITE if ri%2==0 else LGRAY
        y=6.0-ri*0.55
        rbox(ax,0.3,y,13.7,0.5,'',fc=bg,ec=MGRAY,tc=NAVY,fs=8,lw=0.5)
        x=0.3
        for ci,(val,w) in enumerate(zip(row[:6],widths)):
            if ci==3:
                if val=='Open':
                    rbox(ax,x+0.05,y+0.1,1.3,0.3,'Open',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
                elif val=='Closed':
                    rbox(ax,x+0.05,y+0.1,1.3,0.3,'Closed',fc='#fadbd8',ec=RED,tc=RED,fs=7.5)
                elif val=='[dropdown]':
                    rbox(ax,x+0.05,y+0.05,1.4,0.4,'Status v',fc=WHITE,ec=MGRAY,tc=NAVY,fs=7.5)
                    # Dropdown menu
                    rbox(ax,x+0.05,y-0.6,1.4,0.6,'Open\nClosed\nArchived',fc=WHITE,ec=NAVY,tc=NAVY,fs=7,lw=1)
                else:
                    ax.text(x+w/2,y+0.25,val,ha='center',va='center',fontsize=7.5,color=NAVY)
            elif ci==5:
                rbox(ax,x+0.1,y+0.1,0.8,0.3,'...',fc=LGRAY,ec=MGRAY,tc=NAVY,fs=9,bold=True)
            else:
                ax.text(x+0.1,y+0.25,str(val),ha='left',va='center',fontsize=7.5,color=NAVY)
            x+=w+0.05

    # Context menu for row 1
    rbox(ax,12.0,5.45,1.5,0.9,'فتح\nتعديل\nحذف',fc=WHITE,ec=NAVY,tc=NAVY,fs=7.5,lw=1.5)

    # Pagination
    rbox(ax,5.5,3.5,3.0,0.45,'< Prev  1  2  3  Next >',fc=WHITE,ec=MGRAY,tc=NAVY,fs=8.5)

    caption(fig,'Figure 3.5: Dashboard — case management interface')
    save_fig(fig,'fig3_5_dashboard_mockup.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.6
# ─────────────────────────────────────────────────────────────────
def fig3_6():
    fig, ax = plt.subplots(figsize=(13,10))
    ax.set_xlim(0,13); ax.set_ylim(0,10)
    ax.axis('off'); ax.set_facecolor(WHITE)

    # Warning banner
    rbox(ax,0,9.2,13,0.7,'',fc='#d4ac0d',ec='#b7950b')
    ax.text(6.5,9.55,u'⚠ AI assistant only — Judge retains absolute decision authority',
            ha='center',va='center',fontsize=9,color=WHITE,fontweight='bold')

    # Nav bar
    rbox(ax,0,8.5,13,0.65,'',fc=NAVY,ec=NAVY)
    ax.text(0.3,8.82,'Hakim',ha='left',va='center',fontsize=10,color=GOLD,fontweight='bold')
    ax.text(12.7,8.82,'Home   Logout',ha='right',va='center',fontsize=8,color=WHITE)

    # Chat area
    rbox(ax,0.2,1.0,12.6,7.2,'',fc=LGRAY,ec=MGRAY,lw=0.5)

    # User message (right aligned)
    rbox(ax,7.5,6.5,5.0,1.0,'',fc=NAVY,ec=NAVY)
    ax.text(12.3,7.05,u'What are the conditions for\nrescission in binding contracts?',
            ha='right',va='center',fontsize=8,color=WHITE)
    ax.text(7.7,7.0,u'⚖',ha='left',va='center',fontsize=13,color=GOLD)
    ax.text(7.7,6.55,'10:32 AM',ha='left',va='bottom',fontsize=6.5,color=MGRAY)

    # AI typing indicator
    rbox(ax,0.5,5.3,2.5,0.65,'',fc=WHITE,ec=MGRAY)
    for xi in [1.0,1.4,1.8]:
        circle=plt.Circle((xi,5.62),0.12,fc=NAVY,ec=NAVY,zorder=3)
        ax.add_patch(circle)
    ax.text(0.7,5.55,'AI Responding...',ha='left',va='center',fontsize=7,color=MGRAY,style='italic')

    # AI response bubble
    rbox(ax,0.5,2.5,9.0,2.6,'',fc=WHITE,ec=NAVY,lw=1.5)
    ax.text(0.7,5.0,u'\U0001f916',ha='left',va='top',fontsize=12)
    ai_text=(u'Article 157 of the Egyptian Civil Code stipulates that\n'
             u'rescission in bilateral contracts requires: (1) a binding\n'
             u'obligation that has not been fulfilled, (2) a notice of\n'
             u'intent to rescind served on the defaulting party, and\n'
             u'(3) judicial approval unless the contract contains an\n'
             u'explicit rescission clause. See also Articles 158-160.')
    ax.text(1.3,4.85,ai_text,ha='left',va='top',fontsize=8,color=NAVY,linespacing=1.5)
    ax.text(0.7,2.55,'10:33 AM',ha='left',va='bottom',fontsize=6.5,color=MGRAY)

    # Sources chips
    for ci,src in enumerate(['Art. 157','Art. 158','Art. 160']):
        rbox(ax,1.3+ci*1.5,2.55,1.3,0.3,src,fc=LBLUE,ec=BLUE,tc=BLUE,fs=7)

    # Input area
    rbox(ax,0.2,0.2,12.6,0.75,'',fc=WHITE,ec=MGRAY)
    ax.text(0.5,0.57,u'\U0001f4ce',ha='left',va='center',fontsize=11,color=MGRAY)
    rbox(ax,1.0,0.28,10.2,0.55,u'Type your legal query...',fc=LGRAY,ec=MGRAY,tc=MGRAY,fs=8.5)
    rbox(ax,11.3,0.28,1.2,0.55,'Send  →',fc=NAVY,ec=NAVY,tc=WHITE,fs=8,bold=True)

    caption(fig,'Figure 3.6: Case Chat interface with SSE streaming response')
    save_fig(fig,'fig3_6_case_chat_mockup.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.7
# ─────────────────────────────────────────────────────────────────
def fig3_7():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    # Nav bar
    rbox(ax,0,8.2,14,0.7,'',fc=NAVY,ec=NAVY)
    ax.text(0.3,8.55,'Hakim | OCR Viewer',ha='left',va='center',fontsize=10,color=GOLD,fontweight='bold')

    # Left sidebar
    rbox(ax,0,0.8,3.5,7.3,'',fc=LGRAY,ec=MGRAY,lw=1)
    ax.text(1.75,7.9,u'Documents (3)',ha='center',va='center',fontsize=9,color=NAVY,fontweight='bold')

    # Doc items
    items=[
        (u'Court Pleading.pdf',u'✓ Ready',GREEN,True),
        (u'Expert Report.pdf',u'... Processing',GOLD,False),
        (u'Defense Memo.pdf',u'✗ Failed',RED,False),
    ]
    for i,(name,status,sc,active) in enumerate(items):
        y=7.2-i*1.1
        lw=3 if active else 1
        lc=NAVY if active else MGRAY
        rbox(ax,0.05,y-0.35,3.4,0.8,'',fc=WHITE if active else LGRAY,ec=lc,lw=lw)
        if active:
            ax.add_patch(FancyBboxPatch((0.05,y-0.35),0.15,0.8,boxstyle='round,pad=0',fc=NAVY,ec=NAVY))
        ax.text(0.4,y+0.1,name,ha='left',va='center',fontsize=7.5,color=NAVY,fontweight='bold' if active else 'normal')
        ax.text(0.4,y-0.15,status,ha='left',va='center',fontsize=7,color=sc)

    # Right panel
    rbox(ax,3.6,0.8,10.3,7.3,'',fc=WHITE,ec=MGRAY,lw=1)

    # Panel header
    rbox(ax,3.6,7.3,10.3,0.8,'',fc=NAVY,ec=NAVY)
    ax.text(3.9,7.7,u'Extracted Digital Text',ha='left',va='center',fontsize=9.5,color=WHITE,fontweight='bold')
    ax.text(12.5,7.7,u'\U0001f50d  Copy  ✎ Edited',ha='right',va='center',fontsize=8.5,color=GOLD)

    # Metadata
    rbox(ax,3.6,6.8,10.3,0.45,'',fc=LBLUE,ec=LBLUE)
    ax.text(3.9,7.02,u'Word count: 1,245    Last edited: 10:45 AM',ha='left',va='center',fontsize=8,color=NAVY)
    rbox(ax,12.0,6.87,1.6,0.28,u'✎ Edited',fc=GOLD,ec=GOLD,tc=NAVY,fs=7.5)

    # Text content area
    text_content=(
        u'In the Name of Allah, the Most Gracious, the Most Merciful\n\n'
        u'Court Pleading No. 123/2024 Civil\n'
        u'Before the Cairo Court of First Instance\n\n'
        u'The plaintiff, Mohamed Ibrahim Hassan, hereby submits\n'
        u'this pleading against the defendant, Ahmed Mahmoud Ali,\n'
        u'requesting the court to rule in accordance with the\n'
        u'following claims and supporting documentation...\n\n'
        u'[Document continues — 1,245 words total]'
    )
    ax.text(4.0,6.6,text_content,ha='left',va='top',fontsize=8,color=NAVY,linespacing=1.6)

    # Floating toolbar
    rbox(ax,3.5,0.2,7.0,0.55,'',fc=NAVY,ec=NAVY,radius=0.15)
    ax.text(4.2,0.47,u'✎ Edit Texts',ha='left',va='center',fontsize=8.5,color=WHITE,fontweight='bold')
    ax.plot([7.0,7.0],[0.25,0.7],color=MGRAY,lw=1)
    ax.text(7.5,0.47,u'Editing 2 files',ha='left',va='center',fontsize=8,color=MGRAY)
    rbox(ax,9.0,0.3,1.2,0.37,'Cancel',fc='#555',ec='#555',tc=WHITE,fs=7.5)
    rbox(ax,10.3,0.3,1.8,0.37,u'✓ Confirm & Send',fc=GREEN,ec=GREEN,tc=WHITE,fs=7.5,bold=True)

    caption(fig,'Figure 3.7: OCR viewer with multi-document sidebar and edit toolbar')
    save_fig(fig,'fig3_7_ocr_mockup.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.8
# ─────────────────────────────────────────────────────────────────
def fig3_8():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    # Nav bar
    rbox(ax,0,8.2,14,0.7,'',fc=NAVY,ec=NAVY)
    ax.text(0.3,8.55,'Hakim | Legal Search',ha='left',va='center',fontsize=10,color=GOLD,fontweight='bold')

    # Left sidebar
    rbox(ax,0,0.5,3.5,7.6,'',fc=LGRAY,ec=MGRAY)
    ax.text(1.75,7.9,u'Legal Search',ha='center',va='center',fontsize=10,color=NAVY,fontweight='bold')

    # Mode buttons
    rbox(ax,0.1,7.1,1.6,0.45,u'★ AI Query',fc=GOLD,ec=GOLD,tc=NAVY,fs=8,bold=True)
    rbox(ax,1.8,7.1,1.6,0.45,u'# By Article',fc=WHITE,ec=MGRAY,tc=NAVY,fs=8)

    # Form
    ax.text(0.2,6.85,u'Law Corpus:',ha='left',va='center',fontsize=7.5,color=NAVY)
    rbox(ax,0.1,6.45,3.3,0.38,u'Civil Code  ▼',fc=WHITE,ec=MGRAY,tc=NAVY,fs=8)
    ax.text(0.2,6.25,u'Query:',ha='left',va='center',fontsize=7.5,color=NAVY)
    rbox(ax,0.1,5.2,3.3,0.95,u'What are conditions\nfor rescission...?',fc=WHITE,ec=MGRAY,tc=NAVY,fs=7.5)
    rbox(ax,0.8,4.75,1.7,0.38,u'Search',fc=NAVY,ec=NAVY,tc=WHITE,fs=8.5,bold=True)

    ax.plot([0.1,3.4],[4.55,4.55],color=MGRAY,lw=0.8)
    ax.text(0.2,4.35,u'Corpus Browser',ha='left',va='center',fontsize=7.5,color=NAVY,fontweight='bold')
    for i,corpus in enumerate([u'\U0001f4da Civil Code (active)',u'\U0001f4da Evidence Law',u'\U0001f4da Procedures Law']):
        y=4.0-i*0.5
        fc2=LBLUE if i==0 else LGRAY
        rbox(ax,0.1,y-0.18,3.3,0.38,corpus,fc=fc2,ec=MGRAY if i>0 else BLUE,tc=NAVY,fs=7.5)

    # Right panel
    rbox(ax,3.6,0.5,10.3,7.6,'',fc=WHITE,ec=MGRAY)

    # Results header
    rbox(ax,3.6,7.6,10.3,0.6,'',fc=NAVY,ec=NAVY)
    ax.text(3.9,7.9,u'Search Results',ha='left',va='center',fontsize=9.5,color=WHITE,fontweight='bold')
    ax.text(13.7,7.9,u'3 articles found',ha='right',va='center',fontsize=8,color=GOLD)

    # AI Answer card
    rbox(ax,3.8,1.5,10.0,5.8,'',fc=WHITE,ec=MGRAY,lw=1)
    ax.add_patch(FancyBboxPatch((3.8,6.85),10.0,0.4,boxstyle='round,pad=0',fc=GOLD,ec=GOLD))
    ax.add_patch(FancyBboxPatch((3.8,1.5),0.15,5.8,boxstyle='round,pad=0',fc=NAVY,ec=NAVY))
    ax.text(3.95,7.05,u'★ AI Legal Answer — Generated by Judicial Assistant',
            ha='left',va='center',fontsize=8.5,color=NAVY,fontweight='bold')

    ans_text=(u'Based on Article 157 of the Egyptian Civil Code, rescission\n'
              u'of binding bilateral contracts requires three conditions:\n\n'
              u'1. Existence of a binding obligation between the parties\n'
              u'2. Material breach by one of the contracting parties\n'
              u'3. Filing a rescission lawsuit before competent courts\n\n'
              u'The court may grant a grace period if appropriate (Art. 158).\n'
              u'Parties may include an explicit rescission clause (Art. 160).')
    ax.text(4.1,6.75,ans_text,ha='left',va='top',fontsize=8.5,color=NAVY,linespacing=1.55)

    ax.text(4.1,2.3,u'Referenced Articles:',ha='left',va='center',fontsize=8,color=NAVY,fontweight='bold')
    for ci,art in enumerate([u'Art. 157',u'Art. 158',u'Art. 160']):
        rbox(ax,4.1+ci*1.7,1.6,1.5,0.5,art,fc=LBLUE,ec=BLUE,tc=BLUE,fs=8,bold=True)

    caption(fig,'Figure 3.8: Legal Search interface — AI query mode with answer and sources')
    save_fig(fig,'fig3_8_legal_search_mockup.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.9
# ─────────────────────────────────────────────────────────────────
def fig3_9():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def table_box(ax, x, y, w, name, fields):
        hdr_h=0.55; row_h=0.3; body_h=len(fields)*row_h+0.15
        rbox(ax,x,y+body_h,w,hdr_h,name,fc=NAVY,ec=NAVY,tc=WHITE,fs=9,bold=True)
        rbox(ax,x,y,w,body_h,'',fc=WHITE,ec=NAVY,tc=NAVY,fs=7.5,lw=1.5)
        for i,f in enumerate(fields):
            ax.plot([x,x+w],[y+body_h-(i+1)*row_h]*2,color=MGRAY,lw=0.5)
            ax.text(x+0.15,y+body_h-(i+0.5)*row_h,f,ha='left',va='center',fontsize=7,color=NAVY)
        return y+body_h+hdr_h

    # judges
    table_box(ax,0.3,3.5,5.5,'judges',
        [u'\U0001f511 id (PK, UUID)',
         'firstName (varchar)','lastName (varchar)',
         'judicialNumber (UNIQUE)','email (UNIQUE)',
         'phoneNumber (UNIQUE)','courtName (varchar)',
         'nationalId (UNIQUE)','password (hashed)',
         'otpCode / otpExpiry','loginAttempts / lockUntil',
         'tokenVersion','profileImage','createdAt / updatedAt'])

    # admins
    table_box(ax,7.5,3.8,5.8,'admins',
        [u'\U0001f511 id (PK, UUID)',
         'firstName (varchar)','lastName (varchar)',
         'email (UNIQUE)','phoneNumber (UNIQUE)',
         'password (hashed)','otpCode / otpExpiry',
         'loginAttempts / lockUntil',
         "role (DEFAULT 'superadmin')",
         'tokenVersion','createdAt / updatedAt'])

    # SupportTickets
    table_box(ax,0.3,0.4,5.5,'SupportTickets',
        [u'\U0001f511 id (PK, UUID)',
         'status (varchar)',
         'judgeName (varchar)','judgeEmail (varchar)',
         u'\U0001f517 SupportTicketId (FK)',
         'createdAt / updatedAt'])

    # SupportMessages
    table_box(ax,7.5,0.4,5.8,'SupportMessages',
        [u'\U0001f511 id (PK, UUID)',
         u'\U0001f517 SupportTicketId (FK → SupportTickets.id)',
         'content (text)','senderType (varchar)',
         'createdAt'])

    # FK arrow
    ax.annotate('',xy=(7.5,1.5),xytext=(5.8,1.5),
                arrowprops=dict(arrowstyle='->',color=RED,lw=1.5),annotation_clip=False)
    ax.text(6.65,1.65,'CASCADE DELETE, 1..*',ha='center',va='bottom',fontsize=7,color=RED)

    ax.set_title('PostgreSQL Relational Schema — Web Back-End',fontsize=12,color=NAVY,fontweight='bold',pad=10)
    caption(fig,'Figure 3.9: PostgreSQL relational schema (web back-end)')
    save_fig(fig,'fig3_9_postgres_er.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.10
# ─────────────────────────────────────────────────────────────────
def fig3_10():
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_xlim(0,14); ax.set_ylim(0,9)
    ax.axis('off'); ax.set_facecolor(WHITE)

    actors=['Browser','Node.js BFF','PostgreSQL','Email Server\n(SMTP)']
    xs=[1.2,4.5,8.0,12.0]
    cols=[GOLD,NAVY,BLUE,GREEN]

    for name,x,c in zip(actors,xs,cols):
        rbox(ax,x-0.8,8.2,1.6,0.65,name,fc=c,ec=NAVY,tc=WHITE,fs=8.5,bold=True)
        ax.plot([x,x],[0.5,8.2],color=NAVY,lw=1,linestyle='dashed',zorder=1)

    def seq(ax,i0,i1,y,lbl,color=NAVY,dashed=False):
        x0,x1=xs[i0],xs[i1]
        ax.annotate('',xy=(x1,y),xytext=(x0,y),
                    arrowprops=dict(arrowstyle='->',color=color,lw=1.2,linestyle='dashed' if dashed else 'solid'),
                    annotation_clip=False)
        ax.text((x0+x1)/2,y+0.12,lbl,ha='center',va='bottom',fontsize=7.5,color=color)

    # Step 1 auth
    seq(ax,0,1,7.6,'1. POST /api/login {courtName, identifier, password}')
    seq(ax,1,2,7.0,'2. SELECT judge WHERE courtName ILIKE + (phone OR judicialNumber)')
    seq(ax,2,1,6.4,'3. judge record (or 404)',color=GREEN)
    ax.text(4.5,6.15,'3a. bcrypt.compare + loginAttempts check (>=5 = 15min lock)',
            ha='center',va='top',fontsize=7,color=NAVY,style='italic')
    seq(ax,1,1,5.8,'4. generate 6-digit OTP, set otpExpiry = now+5min',color=NAVY)
    ax.annotate('',xy=(4.5,5.8),xytext=(4.5,5.8),
                arrowprops=dict(arrowstyle='-',color=NAVY,lw=1),annotation_clip=False)
    ax.text(4.5,5.75,'[internal]',ha='center',va='top',fontsize=6.5,color=MGRAY,style='italic')
    seq(ax,1,2,5.3,'5. UPDATE otpCode, otpExpiry')
    seq(ax,1,3,4.7,'6. sendMail {to: judge.email, subject: OTP code}')
    seq(ax,1,0,4.1,'7. {preAuthToken (5min JWT), masked email}',color=GOLD)

    # Separator
    ax.plot([0.2,13.8],[3.7,3.7],color=MGRAY,lw=1.5,linestyle='dashed')
    ax.text(7,3.8,'Step 2: OTP Verification',ha='center',va='bottom',fontsize=8.5,color=NAVY,fontweight='bold',style='italic')

    seq(ax,0,1,3.3,'8. POST /api/verify-otp {preAuthToken, otpCode}')
    ax.text(4.5,3.1,'9. jwt.verify(preAuthToken) + otpCode match + otpExpiry > now',
            ha='center',va='top',fontsize=7,color=NAVY,style='italic')
    seq(ax,1,2,2.7,'10. UPDATE otpCode=null, otpExpiry=null, loginAttempts=0')
    ax.text(4.5,2.52,'11. jwt.sign({id, user_id, role, version}, expiresIn:\'8h\')',
            ha='center',va='top',fontsize=7,color=NAVY,style='italic')
    seq(ax,1,0,2.0,"12. Set-Cookie: token=JWT; HttpOnly; SameSite=Lax; 8h",color=GREEN)

    caption(fig,'Figure 3.10: Two-factor authentication flow (password + OTP)')
    save_fig(fig,'fig3_10_auth_flow.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.11
# ─────────────────────────────────────────────────────────────────
def fig3_11():
    fig, ax = plt.subplots(figsize=(16,6))
    ax.set_xlim(0,16); ax.set_ylim(0,6)
    ax.axis('off'); ax.set_facecolor(WHITE)

    stages=[
        ('Stage 1\nIngestion\n\nPDF → rasterize\n@ 400 DPI',LBLUE,NAVY),
        ('Stage 2\nRestoration\n\nCLAHE contrast\nnormalization',LBLUE,NAVY),
        ('Stage 3\nPerspective\nCorrection\n\nCanny edge\nCrop ratio ≥0.65\nArea ≥35%',LBLUE,NAVY),
        ('Stage 4\nOCR Engine\n\nQari-OCR-v0.3\nVL-2B-Instruct\n8-bit quantized\nfloat16 GPU\nmax_tokens=4000',NAVY,WHITE),
        ('Stage 5\nText\nReconstruction\n\nEastern numerals\n→ Western numerals',LBLUE,NAVY),
    ]

    inputs=['PDF/Image','Page images\nList[np.ndarray]','Preprocessed\nimages','Perspective-\ncorrected images','Raw OCR text\nList[str]']
    outputs=['Page images\nList[np.ndarray]','Preprocessed\nimages','Corrected\nimages','Raw OCR text\nList[str]','Normalized text']

    # Input arrow
    ax.annotate('',xy=(0.4,3.0),xytext=(0,3.0),
                arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.5),annotation_clip=False)
    ax.text(0,3.05,'PDF /\nImage',ha='center',va='bottom',fontsize=7.5,color=NAVY)

    for i,(text,fc,tc) in enumerate(stages):
        x=0.5+i*3.0
        rbox(ax,x,1.0,2.6,4.0,text,fc=fc,ec=NAVY,tc=tc,fs=8,lw=2 if i==3 else 1.5,radius=0.1)
        if i<len(stages)-1:
            arrow(ax,x+2.6,3.0,x+3.0,3.0,color=NAVY,lw=1.5)
        # Annotation below
        ax.text(x+1.3,0.8,outputs[i],ha='center',va='top',fontsize=6.5,color=MGRAY,style='italic')

    # actual output
    rbox(ax,13.15,1.2,2.6,3.6,'OCRDocumentResult\n\nList[OCRPageResult]\n{page_num,\nraw_text,\nconfidence,\nerror}',
         fc=GOLD,ec=NAVY,tc=NAVY,fs=8,lw=2,radius=0.1)
    arrow(ax,13.1,3.0,13.15,3.0,color=NAVY,lw=1.5)

    ax.set_title('Five-Stage OCR Processing Pipeline',fontsize=12,color=NAVY,fontweight='bold',pad=8)
    caption(fig,'Figure 3.11: Five-stage OCR processing pipeline')
    save_fig(fig,'fig3_11_ocr_pipeline.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.12
# ─────────────────────────────────────────────────────────────────
def fig3_12():
    fig, ax = plt.subplots(figsize=(14,13))
    ax.set_xlim(0,14); ax.set_ylim(0,13)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def oval(ax,cx,cy,w,h,text,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8):
        ellipse=mpatches.Ellipse((cx,cy),w,h,fc=fc,ec=ec,lw=1.5,zorder=2)
        ax.add_patch(ellipse)
        ax.text(cx,cy,text,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold',zorder=3)

    # START
    oval(ax,7,12.5,1.5,0.5,'START',fc=NAVY,tc=WHITE)
    arrow(ax,7,12.25,7,11.8)

    rbox(ax,5,11.3,4,0.9,'preprocessor_node\n(Arabic ratio ≥30%, len 3-2000)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,7,11.3,7,10.7)

    diamond(ax,7,10.3,3.5,0.8,'[Classification\nRouter]',fc=LGRAY)

    # off_topic branch
    arrow(ax,5.25,10.3,3.0,10.3,label='off-topic',fs=7,labelpad=(-0.5,0.08),color=RED)
    rbox(ax,1.5,9.85,2.5,0.7,'off_topic_node',fc='#fadbd8',ec=RED,tc=RED,fs=8)
    oval(ax,2.75,9.3,1.5,0.4,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,2.75,9.85,2.75,9.5)

    # textual branch
    arrow(ax,7,9.9,7,9.4,label='textual',fs=7,labelpad=(0.05,0.05))
    rbox(ax,5.5,9.0,3,0.7,'textual_node',fc=LBLUE,ec=NAVY,fs=8)
    oval(ax,7,8.5,1.5,0.4,'END',fc=NAVY,tc=WHITE,fs=7)
    arrow(ax,7,9.0,7,8.72)

    # scope branch
    arrow(ax,8.75,10.3,11,10.3,label='scope',fs=7,labelpad=(0.05,0.08),color=GOLD)
    rbox(ax,9.5,9.85,3,0.7,'scope_classifier_node',fc=GOLD,ec=NAVY,tc=NAVY,fs=8,bold=True)
    arrow(ax,11,9.85,11,9.3)

    diamond(ax,11,8.9,3,0.7,'[Cache hit?\ncosine ≥0.97]',fc='#fef9e7')
    arrow(ax,9.5,8.9,8.0,8.9,label='YES',fs=7,labelpad=(-0.9,0.08),color=GREEN)
    rbox(ax,5.5,8.55,2.5,0.6,'return cached\nanswer',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
    oval(ax,6.75,8.05,1.3,0.38,'END',fc=GREEN,tc=WHITE,fs=7)
    arrow(ax,6.75,8.55,6.75,8.24)

    arrow(ax,11,8.55,11,8.0,label='NO',fs=7,labelpad=(0.05,0.05))
    rbox(ax,9.5,7.5,3,0.75,'retrieve_node\n(Qdrant judicial_docs\nk=8, BAAI/bge-m3)',fc=NAVY,ec=NAVY,tc=WHITE,fs=7.5)
    arrow(ax,11,7.5,11,7.0)
    rbox(ax,9.5,6.5,3,0.7,'rule_grader_node',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,11,6.5,11,6.0)

    diamond(ax,11,5.6,3,0.7,'[Grade Router]',fc=LGRAY)

    # pass
    arrow(ax,9.5,5.6,8.0,5.6,label='pass',fs=7,labelpad=(-0.7,0.08),color=GREEN)
    rbox(ax,5.5,5.25,2.5,0.65,'generate_answer_node',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
    oval(ax,6.75,4.75,1.3,0.38,'END',fc=GREEN,tc=WHITE,fs=7)
    arrow(ax,6.75,5.25,6.75,4.94)

    # refine
    arrow(ax,12.5,5.6,13.5,5.6,label='refine',fs=7,color=GOLD)
    rbox(ax,12.8,5.2,1.5,0.6,'refine_node',fc='#fef9e7',ec=GOLD,tc=NAVY,fs=7.5)
    ax.annotate('',xy=(11,7.5),xytext=(13.55,5.2),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2,
                               connectionstyle='arc3,rad=-0.3'),annotation_clip=False)
    ax.text(13.3,6.4,'retry\n≤3',ha='center',fontsize=6.5,color=GOLD,style='italic')

    # llm_grade
    arrow(ax,11,5.25,11,4.75,label='llm_grade',fs=7,labelpad=(0.05,0.05))
    rbox(ax,9.5,4.3,3,0.7,'llm_grader_node\n(llm_call_count ≤ 5)',fc=LBLUE,ec=NAVY,fs=7.5)
    arrow(ax,11,4.3,11,3.8)
    diamond(ax,11,3.4,3,0.7,'[pass/fail]',fc=LGRAY)
    arrow(ax,9.5,3.4,8.0,3.4,label='pass',fs=7,labelpad=(-0.7,0.08),color=GREEN)
    rbox(ax,5.5,3.05,2.5,0.65,'generate_answer_node',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
    oval(ax,6.75,2.55,1.3,0.38,'END',fc=GREEN,tc=WHITE,fs=7)
    arrow(ax,6.75,3.05,6.75,2.74)

    # fail paths → cannot_answer
    arrow(ax,11,3.05,11,2.5,label='fail',fs=7,labelpad=(0.05,0.05),color=RED)
    rbox(ax,9.5,2.0,3,0.7,'cannot_answer_node',fc='#fadbd8',ec=RED,tc=RED,fs=8)
    oval(ax,11,1.5,1.3,0.38,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,11,2.0,11,1.69)
    arrow(ax,12.5,5.6,13.0,5.6,label='',color=RED)

    caption(fig,'Figure 3.12: Civil Law RAG agent graph (LangGraph)')
    save_fig(fig,'fig3_12_civil_law_rag.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.13
# ─────────────────────────────────────────────────────────────────
def fig3_13():
    fig, ax = plt.subplots(figsize=(16,12))
    ax.set_xlim(0,16); ax.set_ylim(0,12)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def oval(ax,cx,cy,w,h,text,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8):
        ellipse=mpatches.Ellipse((cx,cy),w,h,fc=fc,ec=ec,lw=1.5,zorder=2)
        ax.add_patch(ellipse)
        ax.text(cx,cy,text,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold',zorder=3)

    ax.text(8,11.7,'Case Document RAG — Parallel Fan-Out',ha='center',va='center',
            fontsize=13,color=NAVY,fontweight='bold')

    # Main graph
    ax.add_patch(FancyBboxPatch((0.2,5.5),7.5,6.0,boxstyle='round,pad=0,rounding_size=0.1',
                                 fc='#f0f4ff',ec=NAVY,lw=1.5))
    ax.text(0.5,11.3,'Main Graph',ha='left',va='top',fontsize=9,color=NAVY,fontweight='bold')

    oval(ax,4.0,11.0,1.4,0.45,'START',fc=NAVY,tc=WHITE,fs=8)
    rbox(ax,2.8,10.2,2.5,0.65,'questionRewriter',fc=LBLUE,ec=NAVY,fs=8)
    rbox(ax,2.8,9.3,2.5,0.65,'questionClassifier',fc=LBLUE,ec=NAVY,fs=8)
    diamond(ax,4.0,8.5,3.2,0.7,'[onTopic\nRouter]',fc=LGRAY)
    rbox(ax,0.5,8.1,2.2,0.65,'offTopicResponse',fc='#fadbd8',ec=RED,tc=RED,fs=7.5)
    rbox(ax,0.5,7.2,2.2,0.65,'errorResponse',fc='#fadbd8',ec=RED,tc=RED,fs=7.5)
    rbox(ax,2.8,7.2,2.5,0.85,'documentSelector\n(fetch titles,\nclassify mode)',fc=GOLD,ec=NAVY,tc=NAVY,fs=7.5)

    arrow(ax,4.0,11.0,4.0,10.85)
    arrow(ax,4.0,10.2,4.0,9.95)
    arrow(ax,4.0,9.3,4.0,8.85)
    arrow(ax,2.4,8.5,1.6,8.5,label='off',fs=7,color=RED)
    arrow(ax,1.6,8.5,1.6,8.1+0.65)
    arrow(ax,1.6,8.1,1.6,7.85)
    arrow(ax,2.4,8.1,2.8,7.6,color=RED)
    arrow(ax,4.0,8.15,4.0,8.05)

    # Fan-out arrows to branches
    for xi,label in [(8.5,'Branch 1'),(11.0,'Branch 2'),(13.5,'Branch 3')]:
        arrow(ax,5.3,7.6,xi,7.6,color=NAVY,lw=1.5)
        ax.text(xi+0.5,7.7,label,ha='center',fontsize=7,color=NAVY,style='italic')

    rbox(ax,2.8,5.8,2.5,0.65,'mergeAnswers',fc=NAVY,ec=NAVY,tc=WHITE,fs=8)
    oval(ax,4.0,5.3,1.3,0.38,'END',fc=NAVY,tc=WHITE,fs=7)
    arrow(ax,4.0,5.8,4.0,5.5)

    # Merge back arrows
    for xi in [8.5,11.0,13.5]:
        ax.annotate('',xy=(5.3,6.1),xytext=(xi,6.5),
                    arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.2,
                                   connectionstyle='arc3,rad=0.1'),annotation_clip=False)

    # Branch sub-graph
    ax.add_patch(FancyBboxPatch((7.5,0.5),8.3,7.5,boxstyle='round,pad=0,rounding_size=0.1',
                                 fc='#fff8f0',ec=GOLD,lw=2,linestyle='dashed'))
    ax.text(7.7,7.8,'Parallel Sub-Graph (SubQuestionState) — spawned via Send()',
            ha='left',va='top',fontsize=8,color=GOLD,fontweight='bold',style='italic')
    ax.text(7.7,7.5,'sub_answers uses operator.add reducer (concurrent-safe)',
            ha='left',va='top',fontsize=7.5,color=NAVY,style='italic')

    oval(ax,11.5,7.0,1.4,0.45,'START',fc=GOLD,tc=NAVY,fs=8)
    rbox(ax,10.2,6.2,2.5,0.65,'branchDocSelector',fc=LBLUE,ec=NAVY,fs=8)
    diamond(ax,11.5,5.5,3.0,0.65,'[Branch\nRouter]',fc=LGRAY)
    rbox(ax,9.3,4.8,2.2,0.65,'BranchDoc\nFinalizer',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
    rbox(ax,10.2,3.9,2.5,0.8,'retrieve\n(Qdrant case_docs\nfiltered by case_id)',fc=NAVY,ec=NAVY,tc=WHITE,fs=7.5)
    rbox(ax,10.2,2.9,2.5,0.65,'retrievalGrader',fc=LBLUE,ec=NAVY,fs=8)
    diamond(ax,11.5,2.2,3.0,0.65,'[Proceed\nRouter]',fc=LGRAY)
    rbox(ax,10.2,1.3,2.5,0.65,'generateAnswer',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
    rbox(ax,13.3,2.5,2.2,0.65,'refineQuestion',fc='#fef9e7',ec=GOLD,tc=NAVY,fs=7.5)
    rbox(ax,13.3,1.3,2.2,0.65,'cannotAnswer',fc='#fadbd8',ec=RED,tc=RED,fs=7.5)
    oval(ax,11.5,0.7,1.3,0.38,'END',fc=NAVY,tc=WHITE,fs=7)

    arrow(ax,11.5,7.0,11.5,6.85)
    arrow(ax,11.5,6.2,11.5,5.82)
    arrow(ax,11.5,5.82,11.5,5.82)
    arrow(ax,11.5,5.17,11.5,4.7,label='proceed',fs=7,color=GREEN)
    arrow(ax,10.2,5.5,9.3+1.1,5.12,label='finalize',fs=7,color=GREEN)
    arrow(ax,9.3+1.1,4.8,9.3+1.1,4.65)
    arrow(ax,11.5,3.9,11.5,3.55)
    arrow(ax,11.5,2.9,11.5,2.52)
    arrow(ax,11.5,1.87,11.5,1.3+0.65)
    arrow(ax,11.5,1.3,11.5,0.89)
    arrow(ax,13.0,2.2,13.3,2.2,label='refine',fs=7,color=GOLD)
    ax.annotate('',xy=(11.5,3.9),xytext=(14.4,2.5+0.65),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2,
                               connectionstyle='arc3,rad=-0.3'),annotation_clip=False)
    ax.text(14.0,3.4,'retry',ha='center',fontsize=6.5,color=GOLD,style='italic')
    arrow(ax,13.0,2.0,13.3,1.6,label='fail',fs=7,color=RED)

    caption(fig,'Figure 3.13: Case Document RAG — parallel fan-out via Send()')
    save_fig(fig,'fig3_13_case_doc_rag.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.14
# ─────────────────────────────────────────────────────────────────
def fig3_14():
    fig, ax = plt.subplots(figsize=(16,18))
    ax.set_xlim(0,16); ax.set_ylim(0,18)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def oval(ax,cx,cy,w,h,text,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8):
        ellipse=mpatches.Ellipse((cx,cy),w,h,fc=fc,ec=ec,lw=1.5,zorder=2)
        ax.add_patch(ellipse)
        ax.text(cx,cy,text,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold',zorder=3)

    ax.text(8,17.7,'Supervisor LangGraph — 17-Node Orchestration Graph',
            ha='center',va='center',fontsize=14,color=NAVY,fontweight='bold')

    # TOP
    oval(ax,8,17.2,1.5,0.45,'START',fc=NAVY,tc=WHITE)
    arrow(ax,8,16.97,8,16.65)
    rbox(ax,6.2,16.2,3.5,0.7,'validate_input',fc=LBLUE,ec=NAVY,fs=9)
    arrow(ax,8,16.2,8,15.7)
    diamond(ax,8,15.3,3.5,0.7,'[Input Router]',fc=LGRAY)
    # off-topic
    arrow(ax,6.25,15.3,5.0,15.3,label='off-topic',fs=7.5,color=RED)
    rbox(ax,3.0,14.95,2.5,0.65,'off_topic_response',fc='#fadbd8',ec=RED,tc=RED,fs=8)
    oval(ax,4.25,14.4,1.3,0.38,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,4.25,14.95,4.25,14.59)

    # valid path
    arrow(ax,8,14.95,8,14.45,label='valid',fs=7.5,labelpad=(0.05,0.05))
    rbox(ax,6.2,13.95,3.5,0.8,'load_long_term_memory\n(semantic TOP_K=10,\nproc prefs 2000 chars)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,8,13.95,8,13.45)
    rbox(ax,6.2,12.95,3.5,0.75,'classify_intent\n(medium LLM → intent\n+ target_agents)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,8,12.95,8,12.45)
    diamond(ax,8,12.05,3.5,0.7,'[Intent Router]',fc=LGRAY)
    arrow(ax,6.25,12.05,5.0,12.05,label='off-topic',fs=7.5,color=RED)
    rbox(ax,3.0,11.7,2.5,0.65,'off_topic_response',fc='#fadbd8',ec=RED,tc=RED,fs=8)
    oval(ax,4.25,11.2,1.3,0.38,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,4.25,11.7,4.25,11.39)

    arrow(ax,8,11.7,8,11.2,label='routable',fs=7.5,labelpad=(0.05,0.05))
    rbox(ax,6.2,10.7,3.5,0.75,'enrich_context\n(case_summary, doc_titles)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,8,10.7,8,10.1)

    # dispatch_agents tall box
    rbox(ax,5.5,8.3,5.0,2.0,'',fc=NAVY,ec=GOLD,tc=WHITE,fs=8,lw=2.5)
    ax.text(8,10.0,'dispatch_agents',ha='center',va='top',fontsize=9,color=WHITE,fontweight='bold')
    # inner dashed box
    ax.add_patch(FancyBboxPatch((5.7,8.5),4.6,1.4,boxstyle='round,pad=0,rounding_size=0.05',
                                 fc='#1a3a6a',ec=GOLD,lw=1,linestyle='dashed'))
    ax.text(8,9.8,'Tier 0 (ThreadPoolExecutor):',ha='center',va='top',fontsize=7.5,color=GOLD)
    ax.text(8,9.55,'civil_law_rag ∥∥ case_doc_rag',ha='center',va='top',fontsize=7.5,color=WHITE)
    ax.text(8,9.3,'Tier 1 (sequential): ocr, summarize, reason',ha='center',va='top',fontsize=7.5,color=LGRAY)

    arrow(ax,8,8.3,8,7.8)
    diamond(ax,8,7.4,3.5,0.7,'[Post Dispatch]',fc=LGRAY)
    arrow(ax,6.25,7.4,5.0,7.4,label='classify',fs=7,color=NAVY)
    rbox(ax,2.5,7.05,2.8,0.65,'classify_and_store\ndocument',fc=LBLUE,ec=NAVY,fs=7.5)
    arrow(ax,8,7.05,8,6.6,label='merge',fs=7,labelpad=(0.05,0.05))

    rbox(ax,6.0,6.1,4.0,0.8,'merge_responses\n(high LLM, source dedup)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,8,6.1,8,5.65)
    rbox(ax,6.2,5.15,3.5,0.7,'verify_citations',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,8,5.15,8,4.65)
    rbox(ax,6.0,4.1,4.0,0.8,'validate_output\n(low LLM, 4 checks)',fc='#fadbd8',ec=RED,tc=RED,fs=8,bold=True,lw=2)
    arrow(ax,8,4.1,8,3.6)
    diamond(ax,8,3.2,3.5,0.7,'[Validation Router]',fc=LGRAY)

    # pass path
    arrow(ax,9.75,3.2,11.0,3.2,label='pass/partial',fs=7.5,color=GREEN,labelpad=(0.05,0.05))
    for yi,lbl in [(2.7,'update_memory\n(operator.add)'),(2.1,'write_long_term_memory\n(semantic/episodic/proc)'),(1.5,'summarize_history\n(trigger: tokens>=4000)'),(0.9,'audit_log')]:
        rbox(ax,10.5,yi-0.25,4.5,0.5,lbl,fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=7.5)
        if yi>0.9:
            arrow(ax,12.75,yi-0.25,12.75,yi-0.25-0.5+0.5,color=GREEN)

    arrow(ax,12.75,0.9,12.75,0.7,color=GREEN)
    oval(ax,12.75,0.45,1.3,0.38,'END',fc=GREEN,tc=WHITE,fs=7)

    # fail path
    arrow(ax,8,2.85,8,2.4,label='fail',fs=7.5,color=RED,labelpad=(0.05,0.05))
    diamond(ax,8,2.0,3.5,0.65,'[Retries\nexhausted?]',fc='#fef9e7')
    arrow(ax,6.25,2.0,5.0,2.0,label='YES',fs=7,color=RED)
    rbox(ax,2.5,1.65,2.8,0.65,'fallback_response',fc='#fadbd8',ec=RED,tc=RED,fs=8)
    oval(ax,3.9,1.15,1.3,0.38,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,3.9,1.65,3.9,1.34)

    arrow(ax,8,1.67,8,1.2,label='NO,retry\n<=2',fs=7,color=GOLD,labelpad=(0.05,0.05))
    rbox(ax,6.2,0.7,3.5,0.7,'prepare_retry',fc='#fef9e7',ec=GOLD,tc=NAVY,fs=8)
    # Retry loop back arrow in GOLD dashed
    ax.annotate('',xy=(5.5,9.3),xytext=(6.2,1.05),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.8,linestyle='dashed',
                               connectionstyle='arc3,rad=0.4'),annotation_clip=False)
    ax.text(4.0,5.0,'retry loop',ha='center',fontsize=7.5,color=GOLD,style='italic',fontweight='bold',rotation=90)

    caption(fig,'Figure 3.14: Supervisor LangGraph — 17-node orchestration graph')
    save_fig(fig,'fig3_14_supervisor_graph.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.15
# ─────────────────────────────────────────────────────────────────
def fig3_15():
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_xlim(0,14); ax.set_ylim(0,14)
    ax.axis('off'); ax.set_facecolor(WHITE)

    def oval(ax,cx,cy,w,h,text,fc=LGRAY,ec=NAVY,tc=NAVY,fs=8):
        ellipse=mpatches.Ellipse((cx,cy),w,h,fc=fc,ec=ec,lw=1.5,zorder=2)
        ax.add_patch(ellipse)
        ax.text(cx,cy,text,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold',zorder=3)

    ax.text(7,13.7,'Chat Reasoner — Plan-Execute-Replan Loop (LangGraph)',
            ha='center',va='center',fontsize=13,color=NAVY,fontweight='bold')

    oval(ax,7,13.2,1.5,0.45,'START',fc=NAVY,tc=WHITE)
    arrow(ax,7,12.97,7,12.55)
    rbox(ax,5.2,12.0,3.5,0.8,'planner\n(high LLM → List[PlanStep\n{step_id, tool, query, depends_on}])',fc=LBLUE,ec=NAVY,fs=7.5)
    arrow(ax,7,12.0,7,11.5)
    rbox(ax,5.5,11.0,3.0,0.7,'plan_validator',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,7,11.0,7,10.5)
    diamond(ax,7,10.1,3.5,0.7,'[Validator Router]',fc=LGRAY)

    # invalid loop
    arrow(ax,5.25,10.1,4.0,10.1,label='invalid',fs=7,color=RED)
    ax.annotate('',xy=(5.2,12.4),xytext=(4.0,10.1),
                arrowprops=dict(arrowstyle='->',color=RED,lw=1.2,
                               connectionstyle='arc3,rad=0.3'),annotation_clip=False)
    ax.text(3.5,11.3,'retry plan',ha='center',fontsize=6.5,color=RED,style='italic',rotation=90)

    # flawed → replanner
    arrow(ax,7,9.75,7,9.3,label='flawed',fs=7,color=GOLD,labelpad=(0.05,0.05))
    rbox(ax,5.5,8.8,3.0,0.7,'replanner\n(replan_count ≤ 2)',fc='#fef9e7',ec=GOLD,tc=NAVY,fs=7.5)

    # valid → executor_fanout
    arrow(ax,8.75,10.1,10.0,10.1,label='valid',fs=7,color=GREEN)
    rbox(ax,9.5,9.75,3.5,0.65,'executor_fanout\n(Send() → N step_workers)',fc=GREEN,ec=GREEN,tc=WHITE,fs=7.5,bold=True)
    ax.text(9.5,9.6,'PlanStep.tool ∈ {case_doc_rag,\ncivil_law_rag, fetch_summary}',
            ha='left',va='top',fontsize=6.5,color=MGRAY,style='italic')

    # Parallel workers
    for xi,lbl in [(9.5,'step_worker 1'),(11.0,'step_worker 2'),(12.7,'step_worker N')]:
        rbox(ax,xi,8.5,1.3,0.6,lbl,fc=LBLUE,ec=NAVY,fs=7)
        arrow(ax,xi+0.65,9.75,xi+0.65,9.1,color=GREEN)
        arrow(ax,xi+0.65,8.5,xi+0.65,8.1,color=GREEN)

    ax.text(11.0,8.35,'operator.add reducer on step_results',ha='center',fontsize=6.5,color=MGRAY,style='italic')

    rbox(ax,10.0,7.5,3.0,0.65,'collector',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,11.5,7.5,11.5,7.0)
    diamond(ax,11.5,6.6,3.5,0.7,'[Collector Router]',fc=LGRAY)

    # retry step
    arrow(ax,13.25,6.6,13.8,6.6,label='retry',fs=7,color=GOLD)
    ax.annotate('',xy=(13.25,9.75+0.33),xytext=(13.8,6.6),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2,
                               connectionstyle='arc3,rad=-0.3'),annotation_clip=False)
    ax.text(14.0,8.2,'step\nretry',ha='center',fontsize=6.5,color=GOLD,style='italic')

    # synthesize
    arrow(ax,11.5,6.25,11.5,5.75,label='synthesize',fs=7,color=NAVY,labelpad=(0.05,0.05))
    rbox(ax,9.8,5.2,3.5,0.8,'synthesizer\n(high LLM, run_count ≤ 2)',fc=LBLUE,ec=NAVY,fs=8)
    arrow(ax,11.5,5.2,11.5,4.7)
    diamond(ax,11.5,4.3,3.5,0.65,'[Synth Router]',fc=LGRAY)
    arrow(ax,11.5,3.97,11.5,3.5,label='sufficient',fs=7,color=GREEN,labelpad=(0.05,0.05))
    rbox(ax,9.8,3.0,3.5,0.7,'trace_writer',fc='#d5f5e3',ec=GREEN,tc=GREEN,fs=8)
    oval(ax,11.5,2.5,1.3,0.38,'END',fc=GREEN,tc=WHITE,fs=7)
    arrow(ax,11.5,3.0,11.5,2.69)

    # insufficient → replanner
    arrow(ax,9.8,4.3,9.0,4.3,label='insuff.',fs=7,color=GOLD)
    ax.annotate('',xy=(7.0,8.8+0.35),xytext=(9.0,4.3),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2,
                               connectionstyle='arc3,rad=0.3'),annotation_clip=False)

    # replan
    arrow(ax,7,8.8,7,8.35)
    diamond(ax,7,7.95,3.5,0.65,'[Replanner Router]',fc=LGRAY)
    arrow(ax,7,7.62,7,7.1,label='retry',fs=7,color=GOLD,labelpad=(0.05,0.05))
    ax.annotate('',xy=(5.5,11.0+0.35),xytext=(7.0,7.1),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2,
                               connectionstyle='arc3,rad=0.4'),annotation_clip=False)
    ax.text(5.2,9.5,'retry to\nvalidator',ha='center',fontsize=6.5,color=GOLD,style='italic')

    arrow(ax,5.25,7.95,4.0,7.95,label='failed',fs=7,color=RED)
    rbox(ax,2.0,7.6,2.5,0.65,"trace_writer\n(status='failed')",fc='#fadbd8',ec=RED,tc=RED,fs=7.5)
    oval(ax,3.25,7.1,1.3,0.38,'END',fc=RED,tc=WHITE,fs=7)
    arrow(ax,3.25,7.6,3.25,7.29)

    # replan → collector (replan path)
    arrow(ax,9.75,6.6,9.0,6.6,label='replan',fs=7,color=NAVY)
    ax.annotate('',xy=(7.0,8.8+0.35),xytext=(9.0,6.6),
                arrowprops=dict(arrowstyle='->',color=NAVY,lw=1.2,
                               connectionstyle='arc3,rad=-0.3'),annotation_clip=False)

    caption(fig,'Figure 3.15: Chat Reasoner — plan-execute-replan loop (LangGraph)')
    save_fig(fig,'fig3_15_chat_reasoner.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.16
# ─────────────────────────────────────────────────────────────────
def fig3_16():
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_xlim(0,16); ax.set_ylim(0,8)
    ax.axis('off'); ax.set_facecolor(WHITE)

    nodes=[
        ('node_0_intake','Text cleaning\nRTL/tatweel removal\nMetadata extraction\nSegmentation\n→ NormalizedChunk[]',NAVY),
        ('node_1_classify','Role classification\n  plaintiff/defendant\n  expert/court\nArabic gender-aware\nordinal labels\n→ ClassifiedChunk[]',NAVY),
        ('node_2_extract','Bullet extraction\nLegal claims, facts,\narguments per chunk\n→ LegalBullet[]',NAVY),
        ('node_3_aggregate','Group by party role\nAgreed facts\nDisputed facts\nParty positions\n→ RoleAggregation[]',NAVY),
        ('node_4a_cluster','Thematic clustering\nLegal topics\n→ ThemedRole[]',NAVY),
        ('node_4b_synthesize','2-3 para Arabic\nnarrative per theme\n→ RoleThemeSummary[]',NAVY),
        ('node_5_brief','7-section CaseBrief\nrendered_brief\n(Arabic Markdown)\n→ MongoDB (Motor)',GOLD),
    ]

    bw=2.15; gap=0.08; bh_hdr=0.6; bh_body=3.0
    x=0.1

    for i,(hdr,body,hdr_fc) in enumerate(nodes):
        # Header
        rbox(ax,x,4.5,bw,bh_hdr,hdr,fc=hdr_fc,ec=NAVY,tc=WHITE,fs=7.5,bold=True,lw=2 if hdr_fc==GOLD else 1.5)
        # Body
        rbox(ax,x,4.5-bh_body,bw,bh_body,'',fc=LGRAY,ec=NAVY,lw=1.5)
        ax.text(x+0.1,4.3,body,ha='left',va='top',fontsize=7,color=NAVY,linespacing=1.5)
        if i<len(nodes)-1:
            arrow(ax,x+bw,3.2,x+bw+gap+0.01,3.2,color=NAVY,lw=1.5)
        x+=bw+gap

    # Annotation below node_1
    ax.text(2.33,1.3,'Gender-correct ordinals\ne.g. 3rd defendant (masc.)',
            ha='center',va='center',fontsize=6.5,color=NAVY,style='italic',
            bbox=dict(boxstyle='round,pad=0.2',fc='#fffde7',ec=GOLD,alpha=0.9))

    ax.set_title('Summarization Pipeline — 7-Node Arabic Legal Brief Generator',
                 fontsize=12,color=NAVY,fontweight='bold',pad=8)
    caption(fig,'Figure 3.16: Summarization pipeline — 7-node Arabic legal brief generator')
    save_fig(fig,'fig3_16_summarization_pipeline.png')


# ─────────────────────────────────────────────────────────────────
# FIGURE 3.17
# ─────────────────────────────────────────────────────────────────
def fig3_17():
    fig, ax = plt.subplots(figsize=(14,10))
    ax.set_xlim(0,14); ax.set_ylim(0,10)
    ax.axis('off'); ax.set_facecolor(WHITE)

    # TIER 1
    ax.add_patch(FancyBboxPatch((0.2,7.5),13.6,2.1,boxstyle='round,pad=0,rounding_size=0.1',
                                 fc=LBLUE,ec=NAVY,lw=1.5))
    ax.text(0.5,9.45,'Tier 1 — Working Memory (In-Flight)',ha='left',va='top',fontsize=10,color=NAVY,fontweight='bold')
    rbox(ax,1.5,7.7,11.0,1.5,'SupervisorState TypedDict\njudge_query, agent_results, merged_response,\nvalidation_status, retry_count, conversation_history, ...',
         fc=WHITE,ec=NAVY,tc=NAVY,fs=9)
    ax.text(13.5,8.45,'Per-turn, discarded after checkpoint',ha='right',va='center',fontsize=7.5,color=NAVY,style='italic')

    # TIER 2
    ax.add_patch(FancyBboxPatch((0.2,4.2),13.6,2.9,boxstyle='round,pad=0,rounding_size=0.1',
                                 fc=LGRAY,ec=NAVY,lw=1.5))
    ax.text(0.5,6.9,'Tier 2 — Short-Term Memory (Conversation Persistence)',ha='left',va='top',fontsize=10,color=NAVY,fontweight='bold')
    rbox(ax,0.5,4.4,5.5,2.2,'MongoDBSaver\nsupervisor_checkpoints\n(thread_id, checkpoint_id)\nFull state per turn',
         fc=WHITE,ec=NAVY,tc=NAVY,fs=8.5)
    rbox(ax,7.0,4.4,6.5,2.2,'summarize_history_node\nrunning_summary (rolling Arabic)\nSHORT_TERM_KEEP_TURNS = 6\nTrigger: SUMMARIZE_TRIGGER_TOKENS = 4000',
         fc=WHITE,ec=NAVY,tc=NAVY,fs=8.5)

    # TIER 3
    ax.add_patch(FancyBboxPatch((0.2,0.5),13.6,3.4,boxstyle='round,pad=0,rounding_size=0.1',
                                 fc='#ece8f0',ec=NAVY,lw=2))
    ax.text(0.5,3.7,'Tier 3 — Long-Term Memory (Cross-Session Persistence)',ha='left',va='top',fontsize=10,color=NAVY,fontweight='bold')
    ax.text(7.0,3.45,'MongoDBStore → supervisor_memory_store',ha='center',va='top',fontsize=8,color=NAVY,style='italic')

    for ci,(strip_c,title,body) in enumerate([
        (GREEN,'Semantic Facts','(case, case_id, facts)\nTOP_K=10 on load\nSynchronous write'),
        (GOLD,'Episodic Memories','(case, case_id, episodes)\nAsync write\nDelay = 300s'),
        (BLUE,'Procedural Prefs','(user, user_id, prefs)\nAsync write\nInject ≤ 2000 chars'),
    ]):
        bx=0.5+ci*4.5
        rbox(ax,bx,0.7,4.2,2.4,'',fc=WHITE,ec=MGRAY,lw=1)
        ax.add_patch(FancyBboxPatch((bx,0.7),0.2,2.4,boxstyle='round,pad=0',fc=strip_c,ec=strip_c))
        ax.text(bx+0.5,2.85,title,ha='left',va='top',fontsize=8.5,color=NAVY,fontweight='bold')
        ax.text(bx+0.5,2.55,body,ha='left',va='top',fontsize=7.5,color=NAVY,linespacing=1.55)

    # Arrows
    arrow(ax,7,7.5,7,7.1,label='checkpoint after each turn',fs=7.5,labelpad=(0.05,0.05),color=NAVY,lw=1.5)
    arrow(ax,7,4.2,7,3.9,label='reflect (async)',fs=7.5,labelpad=(0.05,0.05),color=NAVY,lw=1.5)
    # Tier 3 → Tier 1 dashed feedback
    ax.annotate('',xy=(0.5,8.0),xytext=(0.5,3.1),
                arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.5,linestyle='dashed',
                               connectionstyle='arc3,rad=0.0'),annotation_clip=False)
    ax.text(0.05,5.5,'load_long_term_memory_node\n(start of each turn)',ha='left',va='center',
            fontsize=7,color=GOLD,style='italic',rotation=90)

    caption(fig,'Figure 3.17: Three-tier memory architecture — working, short-term, and long-term')
    save_fig(fig,'fig3_17_memory_architecture.png')


# ─────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────
figures = [
    ('fig3_1_system_architecture.png', fig3_1),
    ('fig3_2_mongodb_er.png', fig3_2),
    ('fig3_3_request_lifecycle.png', fig3_3),
    ('fig3_4_navigation_flow.png', fig3_4),
    ('fig3_5_dashboard_mockup.png', fig3_5),
    ('fig3_6_case_chat_mockup.png', fig3_6),
    ('fig3_7_ocr_mockup.png', fig3_7),
    ('fig3_8_legal_search_mockup.png', fig3_8),
    ('fig3_9_postgres_er.png', fig3_9),
    ('fig3_10_auth_flow.png', fig3_10),
    ('fig3_11_ocr_pipeline.png', fig3_11),
    ('fig3_12_civil_law_rag.png', fig3_12),
    ('fig3_13_case_doc_rag.png', fig3_13),
    ('fig3_14_supervisor_graph.png', fig3_14),
    ('fig3_15_chat_reasoner.png', fig3_15),
    ('fig3_16_summarization_pipeline.png', fig3_16),
    ('fig3_17_memory_architecture.png', fig3_17),
]

success = []
failed = []

for fname, fn in figures:
    try:
        fn()
        success.append(fname)
    except Exception as e:
        failed.append((fname, str(e)))
        print(f'FAILED {fname}: {e}')

print(f'\n=== SUMMARY ===')
print(f'Generated: {len(success)}/17')
if failed:
    print('FAILED:')
    for f,e in failed:
        print(f'  {f}: {e}')
else:
    print('All figures generated successfully.')
