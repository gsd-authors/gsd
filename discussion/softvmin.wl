(* ExecuteFile["softvmin.wl"] *)

Clear["Global`*"]

fa[x_] :=  (2-x)(x-1)
fb[x_] :=  (3-x)(x-2)

Plot[{fa[x],fb[x]},{x,1,3}] //Export["fafb.pdf", # ]&


pows = {0,2,4}
vars=Subscript[b,#]&/@pows

appf[x_] := Total[Subscript[b,#] (x-2)^# &/@pows]

eqs={
        appf[2-d]==fa[2-d],
        appf[2+d]==fb[2+d],
        D[fa[x],x]==D[appf[x],x]/.{x->2-d},
        D[fb[x],x]==D[appf[x],x]/.{x->2+d},
        D[fa[x],{x,2}]==D[appf[x],{x,2}]/.{x->2-d},
        D[fb[x],{x,2}]==D[appf[x],{x,2}]/.{x->2+d},
        D[appf[x],x]==0/.{x->2}
  }

sol=Solve[
    eqs,
    vars
]

(* sol = vars/.Solve[
    eqs/.{d->0.1},
    vars
] *)

sol=sol[[1]]


Plot[{(appf[x]/.sol)/.{d->1/50}, fa[x],fb[x]},{x,1.8,2.2}, PlotRange->{0,1/4}]//Export["appf.pdf", # ]&

Export["sol.txt",(appf[x]/.sol)]

(* Needs["CCodeGenerator`"]

CCodeGenerator[]


c = Compile[ {{x},{d}}, appf[x]/.sol];
file = CCodeStringGenerate[c, "fun"] *)

(* Test cases *)

(appf[x]/.sol)/.{d->1/50, x->1.99}

(appf[x]/.sol)/.{d->1/10, x->2.05}

p = (n+1)/(n+2)
ep = p x + (1-p)(x+1)

v = Simplify[p (x-ep)^2 + (1-p)(x+1-ep)^2]

ExportString[v,"tex"]

sol = Solve[((appf[x]/.sol)/.{x->2})==v,d]

ExportString[sol,"tex"]

N[(d/.sol[[1]])/.{n->24}]

