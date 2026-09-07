/* One storage location and stale-workspace protection shared by both assays. */
(()=>{
 const originalFetch=window.fetch.bind(window);let current;
 const ready=originalFetch('/api/storage').then(r=>r.json()).then(s=>{current=s;return s;});
 window.fetch=async(url,options={})=>{if((options.method||'GET').toUpperCase()==='POST'){await ready;options={...options,headers:{...options.headers,'X-Workspace':current.workspace_id}};}return originalFetch(url,options);};
 ready.then(s=>{if(!s.available)window.mountStorage(document.querySelector("main"));}).catch(()=>{});
 window.mountStorage=host=>{
  const box=document.createElement('div');box.className='storage-location';const label=document.createElement('span'),button=document.createElement('button');button.textContent='Change location';box.append(label,button);host.prepend(box);
  ready.then(s=>{label.textContent='Storage: '+s.path;label.title=s.path;if(!s.available)label.textContent+=' · Drive disconnected';}).catch(()=>{label.textContent='Storage unavailable';});
  button.onclick=async()=>{
   const dialog=document.createElement('dialog');dialog.className='storage-dialog';dialog.innerHTML='<h2>Analysis location</h2><select aria-label="Drive"></select><div class="storage-path"><input aria-label="Folder path"><button data-go>Go</button><button data-up>Up</button></div><div class="storage-folders"></div><p role="status"></p><div class="storage-actions"><button data-cancel>Cancel</button><button data-use class="primary">Use this folder</button></div>';
   document.body.append(dialog);dialog.showModal();const input=dialog.querySelector('input'),drives=dialog.querySelector('select'),list=dialog.querySelector('.storage-folders'),message=dialog.querySelector('p');let selected;
   const open=async path=>{try{const r=await originalFetch('/api/storage/folders?path='+encodeURIComponent(path||'')),d=await r.json();if(!r.ok)throw Error(d.error);selected=d;input.value=d.path;list.replaceChildren();drives.replaceChildren();for(const drive of d.drives){const o=new Option(drive,drive);drives.add(o);}drives.value=d.drives.find(x=>d.path.startsWith(x))||'';for(const folder of d.folders){const b=document.createElement('button');b.textContent=folder.name;b.onclick=()=>open(folder.path);list.append(b);}message.textContent=d.workspace?'Existing workspace':'A Mouse Behavior Analysis folder will hold videos and results here.';}catch(e){message.textContent=e.message;selected=null;}};
   dialog.querySelector('[data-go]').onclick=()=>open(input.value);input.onkeydown=e=>{if(e.key==='Enter')open(input.value);};drives.onchange=()=>open(drives.value);dialog.querySelector('[data-up]').onclick=()=>open(selected?.parent);
   dialog.querySelector('[data-cancel]').onclick=()=>dialog.close();dialog.onclose=()=>dialog.remove();
   dialog.querySelector('[data-use]').onclick=async()=>{if(!selected)return;const use=dialog.querySelector('[data-use]');use.disabled=true;message.textContent="Preparing the analysis folder…";try{const r=await window.fetch('/api/storage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:selected.path,create_workspace:!selected.workspace})}),d=await r.json();if(!r.ok)throw Error(d.error);location.reload();}catch(e){message.textContent=e.message;use.disabled=false;}};
   await open(current?.available?current.path:'');
  };
 };
})();
