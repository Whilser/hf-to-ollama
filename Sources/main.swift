import Cocoa
import CryptoKit
class Controller: NSObject, NSApplicationDelegate, NSTextFieldDelegate {
 var window: NSWindow!
 let modelsPane = ModelsPane()
 let installedQuant=NSPopUpButton()
 let currentQuant=NSTextField(labelWithString:"Текущее квантование — выбери модель")
 var installedQuantDetails: [String:String] = [:]
 let quantSource=NSTextField()
 let quantName=NSTextField()
 let quantBits=NSPopUpButton()
 let source = NSTextField()
 let chooseSource = NSButton(title:"Открыть скачанную модель…",target:nil,action:nil)
 let name = NSTextField(string: "")
 let status = NSTextField(labelWithString: "Готово к импорту")
 let progress = NSProgressIndicator()
 let log = NSTextView()
 let start = NSButton(title: "Скачать и импортировать", target: nil, action: nil)
 let metrics = NSTextField(labelWithString: "")
 var outputBuffer = ""
 let stop = NSButton(title: "Стоп", target: nil, action: nil)
 var stopping = false
 var process: Process?
 func applicationDidFinishLaunching(_ notification: Notification) {
  let menuBar = NSMenu()
  let applicationItem = NSMenuItem()
  let applicationMenu = NSMenu()
  let quitItem = NSMenuItem(title: "Выйти из HF to Ollama", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
  quitItem.keyEquivalentModifierMask = [.command]
  quitItem.target = NSApp
  applicationMenu.addItem(quitItem)
  applicationItem.submenu = applicationMenu
  menuBar.addItem(applicationItem)
  NSApp.mainMenu = menuBar

  window = NSWindow(contentRect: NSRect(x:0,y:0,width:980,height:850), styleMask:[.titled,.closable,.miniaturizable,.resizable], backing:.buffered,defer:false)
  window.title = "HF to Ollama"
  let stack = NSStackView()
  stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 14
  stack.translatesAutoresizingMaskIntoConstraints = false
  let tabs = NSTabView()
  tabs.translatesAutoresizingMaskIntoConstraints=false
  window.contentView!.addSubview(tabs)
  NSLayoutConstraint.activate([tabs.leadingAnchor.constraint(equalTo:window.contentView!.leadingAnchor,constant:16),tabs.trailingAnchor.constraint(equalTo:window.contentView!.trailingAnchor,constant:-16),tabs.topAnchor.constraint(equalTo:window.contentView!.topAnchor,constant:16),tabs.bottomAnchor.constraint(equalTo:window.contentView!.bottomAnchor,constant:-16)])
  let importTab=NSTabViewItem(identifier:"import");importTab.label="Импорт"
  let container=NSView();container.addSubview(stack);importTab.view=container
  NSLayoutConstraint.activate([stack.leadingAnchor.constraint(equalTo:container.leadingAnchor,constant:16),stack.trailingAnchor.constraint(equalTo:container.trailingAnchor,constant:-16),stack.topAnchor.constraint(equalTo:container.topAnchor,constant:16),stack.bottomAnchor.constraint(equalTo:container.bottomAnchor,constant:-16)])
  tabs.addTabViewItem(importTab)
  let modelsTab=NSTabViewItem(identifier:"models");modelsTab.label="Модели и тесты"
  let modelContainer=NSView()
  let content=modelsPane.view;content.translatesAutoresizingMaskIntoConstraints=false
  modelContainer.addSubview(content)
  NSLayoutConstraint.activate([content.leadingAnchor.constraint(equalTo:modelContainer.leadingAnchor),content.trailingAnchor.constraint(equalTo:modelContainer.trailingAnchor),content.topAnchor.constraint(equalTo:modelContainer.topAnchor),content.bottomAnchor.constraint(equalTo:modelContainer.bottomAnchor)])
  modelsTab.view=modelContainer;tabs.addTabViewItem(modelsTab)
  let quantTab=NSTabViewItem(identifier:"quant");quantTab.label="Квантование"
  let quantPanel=NSStackView();quantPanel.orientation = .vertical;quantPanel.alignment = .leading;quantPanel.spacing=16
  quantPanel.edgeInsets=NSEdgeInsets(top:24,left:24,bottom:24,right:24)
  quantPanel.addArrangedSubview(NSTextField(labelWithString:"Создать копию с другим квантованием"))
  quantPanel.addArrangedSubview(NSTextField(wrappingLabelWithString:"Выбери установленную MLX-модель Ollama или папку Safetensors. Копия получит новое имя и квантование 4 или 8 бит. GGUF пока не поддерживается. Восстановление промежуточных весов требует дополнительного места на диске."))
  quantPanel.addArrangedSubview(NSTextField(labelWithString:"Установленная модель Ollama"))
  installedQuant.target=self;installedQuant.action = #selector(suggestQuantName)
  quantBits.target=self;quantBits.action = #selector(suggestQuantName)
  quantPanel.addArrangedSubview(installedQuant)
  currentQuant.font = .systemFont(ofSize:13)
  currentQuant.textColor = .secondaryLabelColor
  quantPanel.addArrangedSubview(currentQuant)
  quantPanel.addArrangedSubview(NSButton(title:"Обновить список Ollama",target:self,action:#selector(refreshQuantModels)))
  quantPanel.addArrangedSubview(NSButton(title:"Создать копию выбранной модели",target:self,action:#selector(runInstalledQuant)))
  quantSource.placeholderString="Папка с исходными весами Safetensors"
  quantPanel.addArrangedSubview(quantSource)
  quantPanel.addArrangedSubview(NSButton(title:"Выбрать исходную модель…",target:self,action:#selector(chooseQuant)))
  quantBits.addItems(withTitles:["4 бита","8 бит"]);quantPanel.addArrangedSubview(quantBits)
  quantName.placeholderString="Новое имя в Ollama";quantPanel.addArrangedSubview(quantName)
  quantPanel.addArrangedSubview(NSButton(title:"Создать копию",target:self,action:#selector(runQuant)))
  quantPanel.addArrangedSubview(NSTextField(wrappingLabelWithString:"Ход операции и кнопка «Стоп» доступны во вкладке «Импорт»."))
  quantTab.view=quantPanel;tabs.addTabViewItem(quantTab)
  refreshQuantModels()
  modelsPane.canStart = { self.process == nil }
  window.minSize=NSSize(width:820,height:780)
  let title = NSTextField(labelWithString:"Модель из Hugging Face → Ollama")
  title.font = .systemFont(ofSize:24,weight:.semibold)
  stack.addArrangedSubview(title)
  stack.addArrangedSubview(NSTextField(labelWithString:"Ссылка на репозиторий или путь к скачанным весам"))
  source.placeholderString = "https://huggingface.co/автор/модель"
  source.delegate = self
  name.placeholderString = "Имя заполнится по ссылке"
  let paste = NSButton(title: "Вставить из буфера", target: self, action: #selector(pasteSource))
  paste.bezelStyle = .rounded
  paste.setContentHuggingPriority(.required, for: .horizontal)
  paste.setContentCompressionResistancePriority(.required, for: .horizontal)
  let sourceRow = NSStackView(views: [source, paste])
  sourceRow.orientation = .horizontal
  sourceRow.spacing = 8
  stack.addArrangedSubview(sourceRow)
  sourceRow.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
  chooseSource.target=self;chooseSource.action = #selector(folder)
  stack.addArrangedSubview(chooseSource)
  stack.addArrangedSubview(NSButton(title:"Открыть папку со скачанными весами",target:self,action:#selector(openDownloadedWeights)))
  stack.addArrangedSubview(NSTextField(labelWithString:"Имя модели в Ollama"))
  stack.addArrangedSubview(name)
  start.target=self; start.action = #selector(run); start.bezelStyle = .rounded
  stop.target = self; stop.action = #selector(stopWork); stop.isEnabled = false
  let actions = NSStackView(views: [start, stop])
  actions.spacing = 12
  stack.addArrangedSubview(actions)
  stack.addArrangedSubview(status)
  progress.style = .bar; progress.isIndeterminate = true
  stack.addArrangedSubview(progress)
  metrics.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
  stack.addArrangedSubview(metrics)
  let scroll = NSScrollView()
  scroll.hasVerticalScroller = true; scroll.borderType = .bezelBorder
  log.isEditable = false; log.font = .monospacedSystemFont(ofSize:12,weight:.regular)
  log.isVerticallyResizable = true; log.autoresizingMask = [.width]
  scroll.documentView = log
  stack.addArrangedSubview(scroll)
  for view in [name,progress,scroll] as [NSView] { view.widthAnchor.constraint(equalTo:stack.widthAnchor).isActive=true }
  scroll.heightAnchor.constraint(greaterThanOrEqualToConstant:200).isActive=true
  window.center(); window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps:true)
 }
 @objc func refreshQuantModels() {
  guard let url=URL(string:"http://127.0.0.1:11434/api/tags") else{return}
  URLSession.shared.dataTask(with:url) { data,_,error in
   DispatchQueue.main.async {
    if let data=data, let json=try? JSONSerialization.jsonObject(with:data) as? [String:Any],let models=json["models"] as? [[String:Any]] {
     self.installedQuantDetails = [:]
     for model in models {
      guard let name=model["name"] as? String else {continue}
      let details=model["details"] as? [String:Any]
      let quant=(details?["quantization_level"] as? String ?? "").trimmingCharacters(in:.whitespacesAndNewlines)
      self.installedQuantDetails[name] = quant.isEmpty ? "не указано Ollama" : quant
     }
     self.installedQuant.removeAllItems()
     self.installedQuant.addItems(withTitles:models.compactMap{$0["name"] as? String})
     self.suggestQuantName()
    } else { self.status.stringValue=error?.localizedDescription ?? "Не удалось получить список Ollama" }
   }
  }.resume()
 }
 @objc func suggestQuantName() {
  guard let model=installedQuant.titleOfSelectedItem else {
   currentQuant.stringValue="Текущее квантование — модель не выбрана"
   return
  }
  currentQuant.stringValue="Текущее квантование: " + (installedQuantDetails[model] ?? "не указано Ollama")
  quantName.stringValue=model.split(separator:":").first!.replacingOccurrences(of:"/",with:"-")+"-requant:"+(quantBits.indexOfSelectedItem == 0 ? "4bit":"8bit")
 }
 @objc func runInstalledQuant() {
  guard let model=installedQuant.titleOfSelectedItem else{return}
  if quantName.stringValue.isEmpty {
   quantName.stringValue=model.split(separator:":").first!.replacingOccurrences(of:"/",with:"-")+"-requant:"+(quantBits.indexOfSelectedItem == 0 ? "4bit":"8bit")
  }
  launchImport(["python3",Bundle.main.resourcePath!+"/ollama-requant.py","--model",model,"--name",quantName.stringValue,"--bits",quantBits.indexOfSelectedItem == 0 ? "4":"8"])
  if process != nil,let tabs=window.contentView?.subviews.first as? NSTabView {tabs.selectTabViewItem(at:0)}
 }
 func ollamaStorage() -> URL {
  if let path=ProcessInfo.processInfo.environment["OLLAMA_MODELS"], !path.isEmpty {
   return URL(fileURLWithPath:(path as NSString).expandingTildeInPath)
  }
  return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".ollama/models")
 }
 @objc func openOllamaStorage() {
  NSWorkspace.shared.open(ollamaStorage())
 }
 @objc func chooseQuant() {
  let panel=NSOpenPanel();panel.canChooseDirectories=true;panel.canChooseFiles=false
  panel.directoryURL=ollamaStorage()
  panel.message="Выбери исходные Safetensors. Служебные blobs Ollama пока не поддерживаются."

  if panel.runModal() == .OK {quantSource.stringValue=panel.url!.path;quantName.stringValue=panel.url!.lastPathComponent.lowercased()+"-requant"}
 }
 @objc func runQuant() {
  guard !quantSource.stringValue.isEmpty,!quantName.stringValue.isEmpty else {return}
  launchImport(["python3",Bundle.main.resourcePath!+"/requantize.py","--source",quantSource.stringValue,"--name",quantName.stringValue,"--bits",quantBits.indexOfSelectedItem == 0 ? "4":"8"])
  if process != nil, let tabs=window.contentView?.subviews.first as? NSTabView {tabs.selectTabViewItem(at:0)}
 }
 @objc func openDownloadedWeights() {
  let root=FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/HF to Ollama")
  var destination=root
  if let repository=Self.repositoryID(source.stringValue) {
   let digest=SHA256.hash(data:Data((repository+"@main").utf8))
   let key=digest.map {String(format:"%02x",$0)}.joined().prefix(20)
   let candidate=root.appendingPathComponent(String(key))
   if FileManager.default.fileExists(atPath:candidate.path) {destination=candidate}
   else {status.stringValue="Папка этой модели ещё не создана. Открываю общий каталог."}
  } else if source.stringValue.hasPrefix("/") {
   destination=URL(fileURLWithPath:source.stringValue)
  }
  do {
   if destination == root {try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)}
   if !NSWorkspace.shared.open(destination) {status.stringValue="Не удалось открыть папку"}
  } catch {status.stringValue=error.localizedDescription}
 }
 @objc func pasteSource() {
  guard process == nil else { return }
  guard let text = NSPasteboard.general.string(forType: .string),
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
   status.stringValue = "В буфере обмена нет текста"
   return
  }
  source.stringValue = text.trimmingCharacters(in: .whitespacesAndNewlines)
  controlTextDidChange(Notification(name: NSControl.textDidChangeNotification, object: source))
  status.stringValue = "Адрес вставлен"
 }
 func controlTextDidChange(_ notification: Notification) {
  guard notification.object as? NSTextField === source else { return }
  let repository = Self.repositoryID(source.stringValue)
  chooseSource.isHidden = repository != nil
  guard let repository = repository else { return }
  name.stringValue = repository.split(separator: "/")[1].lowercased()
 }
 static func repositoryID(_ input: String) -> String? {
  let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
  var path = text
  if text.contains("://") {
   guard let url = URL(string: text), url.scheme == "https",
         url.host?.lowercased() == "huggingface.co" else { return nil }
   path = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
  }
  guard path.range(of: #"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"#,
                   options: .regularExpression) != nil else { return nil }
  return path
 }
 @objc func folder() {
  let panel=NSOpenPanel(); panel.canChooseDirectories=true; panel.canChooseFiles=false
  if panel.runModal() == .OK { source.stringValue=panel.url!.path; chooseSource.isHidden=false }
 }
 func receive(_ text: String) {
  outputBuffer += text.replacingOccurrences(of: "\r", with: "\n")
  while let end = outputBuffer.firstIndex(of: "\n") {
   let line = String(outputBuffer[..<end])
   outputBuffer = String(outputBuffer[outputBuffer.index(after: end)...])
   if let marker = line.range(of: "@@PROGRESS "),
      let data = String(line[marker.upperBound...]).data(using: .utf8),
      let values = try? JSONSerialization.jsonObject(with: data) as? [String: Double] {
    let done = values["done"] ?? 0
    let total = values["total"] ?? 0
    let speed = values["speed"] ?? 0
    if total > 0 {
     progress.stopAnimation(nil)
     progress.isIndeterminate = false
     progress.doubleValue = min(100, done / total * 100)
    }
    let remaining = max(0, total - done)
    let eta = speed > 0 && total > 0 ? "≈ \(Int(ceil(remaining / speed / 60))) мин" : "время уточняется"
    metrics.stringValue = String(format: "%.2f / %.2f ГБ · осталось %.2f ГБ · %.1f МБ/с · ", done/1e9,total/1e9,remaining/1e9,speed/1e6) + eta
    status.stringValue = speed > 0 ? "Скачивание весов" : "Ожидание данных"
   } else if !line.isEmpty {
    append(line + "\n")
   }
  }
 }
 func append(_ text:String) {
  let clean=text.replacingOccurrences(of:"\\x1B\\[[0-?]*[ -/]*[@-~]",with:"",options:.regularExpression)
  log.textStorage?.append(NSAttributedString(string:clean))
  log.scrollToEndOfDocument(nil)
  if clean.contains("Скачиваю") {status.stringValue="Загрузка модели"}
  if clean.contains("Объединяю") {status.stringValue="Подготовка весов экспертов"}
  if clean.contains("Преобразую") {status.stringValue="Преобразование слоёв"}
  if clean.contains("Импортирую") {status.stringValue="Импорт и проверка в Ollama"; progress.isIndeterminate=true; progress.startAnimation(nil)}
 }
 @objc func run() { launchImport(nil) }
 func launchImport(_ custom: [String]?) {
  guard modelsPane.task == nil, process == nil else {status.stringValue="Сначала останови текущую операцию";return}
  let input=source.stringValue.trimmingCharacters(in:.whitespacesAndNewlines)
  guard custom != nil || (!input.isEmpty && !name.stringValue.isEmpty) else {status.stringValue="Укажи источник и имя модели"; return}
  log.string=""; stopping=false; stop.isEnabled=true; start.isEnabled=false; source.isEnabled=false; name.isEnabled=false
  status.stringValue="Подготовка"; metrics.stringValue=""; outputBuffer=""; progress.isIndeterminate=true; progress.startAnimation(nil)
  let task=Process(); let pipe=Pipe()
  task.executableURL=URL(fileURLWithPath:"/usr/bin/env")
  task.arguments=custom ?? ["python3",Bundle.main.resourcePath!+"/import-model.py",input.hasPrefix("/") ? "--local":"--repo",Self.repositoryID(input) ?? input,"--name",name.stringValue]
  var env=ProcessInfo.processInfo.environment
  env["PATH"]="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
  env["PYTHONUNBUFFERED"]="1"
  task.environment=env; task.standardOutput=pipe; task.standardError=pipe
  pipe.fileHandleForReading.readabilityHandler={ handle in
   let data=handle.availableData
   guard !data.isEmpty else {handle.readabilityHandler=nil; return}
   let text=String(decoding:data,as:UTF8.self)
   DispatchQueue.main.async {self.receive(text)}
  }
  task.terminationHandler={ p in DispatchQueue.main.async {
   self.append(self.outputBuffer); self.outputBuffer=""; self.progress.stopAnimation(nil); self.start.isEnabled=true; self.source.isEnabled=true; self.name.isEnabled=true
   self.stop.isEnabled=false
   self.status.stringValue=self.stopping ? "Остановлено" : p.terminationStatus == 0 ? "Готово — модель отвечает в Ollama":"Ошибка — подробности в журнале"
   self.process=nil
  }}
  do {try task.run(); process=task} catch {
   stop.isEnabled=false; progress.stopAnimation(nil); start.isEnabled=true; source.isEnabled=true; name.isEnabled=true
   append(error.localizedDescription); status.stringValue="Не удалось запустить импорт"
  }
 }
 @objc func stopWork() {
  guard let task = process, task.isRunning else { return }
  stopping = true
  stop.isEnabled = false
  status.stringValue = "Останавливаю…"
  let pid = task.processIdentifier
  if getpgid(pid) == pid {
   kill(-pid, SIGTERM)
  } else {
   task.terminate()
  }
  DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
   if self.process === task && task.isRunning {
    if getpgid(pid) == pid { kill(-pid, SIGKILL) }
    else { kill(pid, SIGKILL) }
   }
  }
 }
 func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {
  if process != nil || modelsPane.task != nil {
   let a=NSAlert(); a.messageText="Остановить работу и выйти?"
   a.addButton(withTitle:"Остановить");a.addButton(withTitle:"Продолжить работу")
   if a.runModal() == .alertFirstButtonReturn {
    stopWork();modelsPane.stopTask()
    Timer.scheduledTimer(withTimeInterval:0.3,repeats:true) { timer in
     if self.process == nil && self.modelsPane.task == nil {
      timer.invalidate();NSApp.reply(toApplicationShouldTerminate:true)
     }
    }
    return .terminateLater
   }
   return .terminateCancel
  }
  return .terminateNow
 }
 func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
}
let app=NSApplication.shared
if let path = Bundle.main.path(forResource: "AppIcon", ofType: "icns"), let icon = NSImage(contentsOfFile: path) { app.applicationIconImage = icon }
let delegate=Controller()
app.delegate=delegate
app.setActivationPolicy(.regular)
app.run()
