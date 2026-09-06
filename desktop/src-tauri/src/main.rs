#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{fs, process::{Child, Command, Stdio}, sync::Mutex, time::{Duration, Instant}};
use tauri::Manager;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};
struct Engine(Mutex<Option<Child>>);
fn stop(app: &tauri::AppHandle) {
    if let Some(mut child) = app.state::<Engine>().0.lock().unwrap().take() {
        // The engine and all analysis/FFmpeg workers share this process group.
        #[cfg(unix)]
        unsafe { libc::kill(-(child.id() as i32), libc::SIGTERM); }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let _=Command::new("taskkill.exe").args(["/PID",&child.id().to_string(),"/T","/F"]).creation_flags(0x08000000).status();
            let _=child.kill();
        }
        let _ = child.wait();
        if let Ok(data)=app.path().app_data_dir() {let _=fs::remove_file(data.join("workspace/.desktop-active"));}
    }
}
fn launch(app: tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let resources = app.path().resource_dir()?.join("bundle");
    let data = app.path().app_data_dir()?;
    fs::create_dir_all(&data)?;
    let workspace = data.join("workspace");
    let ready = data.join(format!("ready-{}.json",std::process::id()));
    let log = fs::OpenOptions::new().create(true).append(true).open(data.join("desktop.log"))?;
    let runtime=if cfg!(windows) {"runtime/python.exe"} else {"runtime/bin/python3"};
    let mut command = Command::new(resources.join(runtime));
    command.arg("-I").arg("-B").arg(resources.join("payload/desktop/service.py"))
        .arg("--bundle").arg(&resources).arg("--workspace").arg(&workspace).arg("--ready").arg(&ready)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONDONTWRITEBYTECODE", "1").stdout(Stdio::from(log.try_clone()?)).stderr(Stdio::from(log));
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    #[cfg(unix)]
    unsafe { command.pre_exec(|| { if libc::setsid() < 0 {return Err(std::io::Error::last_os_error());} Ok(()) }); }
    #[cfg(windows)]
    {use std::os::windows::process::CommandExt; command.creation_flags(0x08000000);}
    let child = command.spawn()?;
    *app.state::<Engine>().0.lock().unwrap() = Some(child);
    let started = Instant::now();
    while !ready.exists() {
        if started.elapsed()>Duration::from_secs(180) { return Err("Analysis engine startup timed out. See desktop.log in the application data folder.".into()); }
        if let Some(child) = app.state::<Engine>().0.lock().unwrap().as_mut() {
            if let Some(code)=child.try_wait()? {return Err(format!("Analysis engine exited ({code}). See {}",data.join("desktop.log").display()).into());}
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    let state:serde_json::Value=serde_json::from_slice(&fs::read(&ready)?)?;
    fs::remove_file(&ready)?;
    let url=state["url"].as_str().ok_or("Missing service URL")?.parse()?;
    tauri::WebviewWindowBuilder::new(&app,"main",tauri::WebviewUrl::External(url))
        .title("Behavior Studio").inner_size(1280.,850.).min_inner_size(900.,650.)
        .on_download(|webview,event| {
            // WebKit's callback runs on the UI thread: do not block on a save dialog.
            // It chooses Downloads and a non-overwriting name before invoking us.
            if let tauri::webview::DownloadEvent::Finished{success,..}=event {
                let text=if success {"Saved to your Downloads folder."} else {"The download could not finish. Please try again."};
                webview.app_handle().dialog().message(text).title("Behavior Studio export").show(|_| {});
            }
            true
        })
        .on_navigation(|url| url.host_str()==Some("127.0.0.1"))
        .build()?;
    if let Some(window)=app.get_webview_window("startup") {window.close()?;}
    Ok(())
}
fn main() {
    let app=tauri::Builder::default().plugin(tauri_plugin_dialog::init())
        .manage(Engine(Mutex::new(None)))
        .on_window_event(|window,event| {
            if window.label()=="main" {
                if let tauri::WindowEvent::CloseRequested{api,..}=event {
                    let h=window.app_handle().clone();
                    let active=h.path().app_data_dir().ok().map(|p|p.join("workspace/.desktop-active").exists()).unwrap_or(false);
                    if active {
                        api.prevent_close();
                        let quit_handle=h.clone();
                        h.dialog().message("Analysis is running. Quit and stop it? Saved results will be kept.").title("Quit Behavior Studio?")
                            .buttons(MessageDialogButtons::OkCancel).show(move |quit| {if quit {stop(&quit_handle);quit_handle.exit(0);}});
                    }
                }
            }
        })
        .setup(|app| {
            let handle=app.handle().clone();
            std::thread::spawn(move || {
                if let Err(error)=launch(handle.clone()) {
                    stop(&handle);
                    handle.dialog().message(error.to_string()).title("Unable to start Behavior Studio").blocking_show();
                    handle.exit(1);
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!()).expect("Unable to create desktop application");
    app.run(|handle,event| {
        if let tauri::RunEvent::ExitRequested{ref api,..}=event {
            // Keep the event loop responsive while the native confirmation is open.
            let active=handle.path().app_data_dir().ok().map(|p|p.join("workspace/.desktop-active").exists()).unwrap_or(false);
            if active {
                api.prevent_exit();
                let h=handle.clone();
                handle.dialog().message("Analysis is running. Quit and stop it? Saved results will be kept.")
                    .title("Quit Behavior Studio?").buttons(MessageDialogButtons::OkCancel).show(move |quit| {if quit {stop(&h);h.exit(0);}});
            } else {stop(handle);}
        }
        if let tauri::RunEvent::Exit=event {stop(handle);}
    });
}
