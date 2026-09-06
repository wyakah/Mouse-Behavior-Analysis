/* Shared upload and mouse metadata setup for every behavioral test. */
class RecordingSetup {
  constructor(host, options) {
    this.host=host;this.options=options;this.busy=false;this.available=[];
    host.innerHTML=`<div class="recording-upload"><label class="button primary">Upload videos<input class="visually-hidden" type="file" multiple accept=".mp4,.avi,.mov,.mkv,.m4v"></label><span data-count></span></div><div class="recording-list"></div><details class="recording-existing"><summary>Choose videos already on this computer</summary><div class="existing-picker"><select aria-label="Workspace video"><option value="">Choose a video</option></select><button data-add>Add video</button></div></details><div class="recording-next"><p data-status role="status"></p><button data-continue class="primary">Continue →</button></div>`;
    this.input=host.querySelector('input[type=file]');this.input.onchange=()=>this.upload([...this.input.files]);
    host.querySelector('[data-add]').onclick=()=>this.addExisting();
    host.querySelector('[data-continue]').onclick=async()=>{try{if(this.valid()&&!this.busy&&!options.disabled?.())await options.continue();}catch(e){this.status(e.message,true);}};
    this.render();
  }
  entries(){return this.options.entries();}
  stranger(e){return e.stranger_side??e.config?.stranger_side??e.config?.target_side??'';}
  valid(){const ids=this.entries().map(e=>String(e.id||'').trim());return ids.length>0&&ids.length<=20&&ids.every(Boolean)&&new Set(ids).size===ids.length&&(!this.options.strangerPosition||this.entries().every(e=>['left','right'].includes(this.stranger(e))));}
  status(message,error=false){const p=this.host.querySelector('[data-status]');p.textContent=message;p.classList.toggle('error',error);}
  update(){
    const n=this.entries().length,disabled=this.busy||this.options.disabled?.();
    this.host.querySelector('[data-count]').textContent=`${n} / 20 videos`;
    this.input.disabled=disabled||n>=20;this.host.querySelector('[data-add]').disabled=disabled||n>=20||!this.available.some(v=>!this.entries().some(e=>e.video===v.name));
    this.host.querySelector('[data-continue]').disabled=disabled||!this.valid();
    if(!this.busy)this.status(!n?'':!this.valid()?(this.options.strangerPosition?'Enter unique mouse IDs and choose each stranger position.':'Enter a unique mouse ID for each video.'):'');
  }
  setAvailable(videos){this.available=videos;this.renderPicker();this.update();}
  renderPicker(){const select=this.host.querySelector('.existing-picker select');select.replaceChildren(new Option('Choose a video',''));for(const v of this.available)if(!this.entries().some(e=>e.video===v.name))select.add(new Option(v.label||v.name.split('/').pop(),v.name));}
  render(){
    const box=this.host.querySelector('.recording-list');box.replaceChildren();
    if(this.entries().length){
      const table=document.createElement('table');table.className='recording-table';const head=table.createTHead().insertRow();for(const label of ['Video','Mouse ID','Sex','Genotype',...(this.options.strangerPosition?['Stranger position']:[]),'']){const th=document.createElement('th');th.textContent=label;th.scope='col';head.append(th);}
      const body=table.createTBody();this.entries().forEach((e,index)=>{
        const row=body.insertRow(),video=row.insertCell();video.textContent=e.video.split('/').pop();video.title=e.video;video.className='recording-name';
        for(const [key,label] of [['id','Mouse ID'],['sex','Sex'],['genotype','Genotype'],...(this.options.strangerPosition?[['stranger_side','Stranger position']]:[])]){
          const cell=row.insertCell(),input=document.createElement(['sex','stranger_side'].includes(key)?'select':'input');cell.dataset.label=label;
          if(key==='sex')for(const [value,text] of [['unknown','Not set'],['female','Female'],['male','Male']])input.add(new Option(text,value));
          else if(key==='stranger_side')for(const [value,text] of [['','Choose side'],['left','Left'],['right','Right']])input.add(new Option(text,value));
          else{input.maxLength=key==='id'?100:120;input.placeholder=key==='id'?'Mouse ID':'Not recorded';}
          input.value=key==='stranger_side'?this.stranger(e):e[key]||(key==='sex'?'unknown':'');input.setAttribute('aria-label',`${label} for video ${index+1}`);input.disabled=this.busy||this.options.disabled?.();
          input.oninput=()=>{e[key]=input.value;if(key==='stranger_side'&&e.config){e.config.stranger_side=input.value;e.config.target_side=input.value;}this.update();Promise.resolve(this.options.change()).catch(error=>this.status(error.message,true));};cell.append(input);
        }
        const remove=document.createElement('button');remove.textContent='×';remove.setAttribute('aria-label',`Remove video ${index+1}`);remove.disabled=this.busy||this.options.disabled?.();remove.onclick=async()=>{this.entries().splice(index,1);this.render();try{await this.options.change();}catch(error){this.status(error.message,true);}};row.insertCell().append(remove);
      });box.append(table);
    }
    this.renderPicker();this.update();
  }
  async add(name){
    if(this.entries().length>=20)throw Error('A setup can contain up to 20 videos. Remove a video to add another.');
    if(this.entries().some(e=>e.video===name))return;
    const e=await this.options.seed(name),base=String(e.id||'Mouse').slice(0,96);let id=base,n=2;while(this.entries().some(e=>e.id===id))id=`${base}-${n++}`;
    this.entries().push({...e,id,sex:e.sex||'unknown',genotype:e.genotype||''});await this.options.change();
  }
  async addExisting(){const select=this.host.querySelector('.existing-picker select');if(!select.value||this.busy)return;const name=select.value;let failure;this.busy=true;this.render();try{await this.add(name);}catch(error){failure=error.message;}finally{this.busy=false;this.render();if(failure)this.status(failure,true);}}
  async upload(files){
    if(this.busy||!files.length)return;
    if(files.length+this.entries().length>20){this.input.value='';this.status(`Choose at most ${20-this.entries().length} more videos. No files were uploaded.`,true);return;}
    this.busy=true;this.options.busy?.(true);this.render();const failures=[];
    try{for(const [i,file] of files.entries()){
      this.status(`Uploading ${i+1} of ${files.length}: ${file.name}`);
      try{const form=new FormData();form.append('file',file);const response=await fetch('/api/videos/upload',{method:'POST',body:form});const data=await response.json();if(!response.ok)throw Error(data.error||'Upload failed.');await this.add(data.name);}catch(error){failures.push(`${file.name}: ${error.message}`);}
    }}finally{this.busy=false;this.options.busy?.(false);this.input.value='';this.render();if(failures.length)this.status(failures.join(' · '),true);}
  }
}
window.RecordingSetup=RecordingSetup;
