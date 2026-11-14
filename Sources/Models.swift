import Cocoa
class ModelsPane: NSObject {
 let view=NSStackView()
 let picker=NSPopUpButton()
 let test=NSButton(title:"Тест моделей",target:nil,action:nil)
 let stop=NSButton(title:"Стоп",target:nil,action:nil)
 let state=NSTextField(labelWithString:"Обнови список моделей")
 let bars=NSStackView()
 let metric = NSSegmentedControl(labels: ["Скорость генерации", "Время загрузки", "Память"], trackingMode: .selectOne, target: nil, action: nil)
 let chartHint = NSTextField(labelWithString: "")
 let chartScroll = NSScrollView()

 let journal=NSTextView()
 var task:Process?
 var pending=""
 var results:[[String:Any]]=[]
 var buttons:[NSButton]=[]
 var canStart:()->Bool = {true}
 override init() {
  super.init()
  view.orientation = .vertical; view.alignment = .leading; view.spacing=16
  view.edgeInsets = NSEdgeInsets(top:20,left:20,bottom:20,right:20)

  let refresh=NSButton(title:"Обновить",target:self,action:#selector(refreshList))
  let rename=NSButton(title:"Переименовать…",target:self,action:#selector(renameModel))
  let delete=NSButton(title:"Удалить…",target:self,action:#selector(deleteModel))
  test.target=self;test.action = #selector(testModels)
  stop.target=self;stop.action = #selector(stopTask);stop.isEnabled=false
  buttons=[refresh,rename,delete,test]
  for v in [NSTextField(labelWithString:"Установленные модели Ollama"),picker,NSStackView(views:[refresh,rename,delete]),NSStackView(views:[test,stop]),state] as [NSView] {view.addArrangedSubview(v)}
  let note=NSTextField(wrappingLabelWithString:"Тест последовательно выгружает и загружает модели. Контекст 4096, до 128 токенов, два запроса. Память — размер загруженной модели по Ollama /api/ps. Скорость — среднее токенов/с. Дисковый кэш macOS влияет на загрузку.")
  note.font = .systemFont(ofSize:12);note.textColor = .secondaryLabelColor;view.addArrangedSubview(note)
  bars.orientation = .vertical;bars.alignment = .leading;bars.spacing=8
  metric.selectedSegment=0;metric.target=self;metric.action = #selector(changeMetric)
  view.addArrangedSubview(metric)
  chartHint.font = .systemFont(ofSize:12);chartHint.textColor = .secondaryLabelColor
  view.addArrangedSubview(chartHint)
  chartScroll.hasVerticalScroller=true
  let canvas=FlippedChartView()
  bars.translatesAutoresizingMaskIntoConstraints=false
  canvas.addSubview(bars)
  chartScroll.documentView=canvas
  canvas.translatesAutoresizingMaskIntoConstraints=false
  NSLayoutConstraint.activate([
   canvas.widthAnchor.constraint(equalTo:chartScroll.contentView.widthAnchor),
   bars.leadingAnchor.constraint(equalTo:canvas.leadingAnchor,constant:4),
   bars.trailingAnchor.constraint(equalTo:canvas.trailingAnchor,constant:-16),
   bars.topAnchor.constraint(equalTo:canvas.topAnchor,constant:8),
   bars.bottomAnchor.constraint(equalTo:canvas.bottomAnchor,constant:-8)])
  view.addArrangedSubview(chartScroll)
  chartScroll.heightAnchor.constraint(greaterThanOrEqualToConstant:220).isActive=true
  let logTitle=NSTextField(labelWithString:"Журнал действий")
  logTitle.font = .systemFont(ofSize:12,weight:.semibold)
  view.addArrangedSubview(logTitle)

  let scroll=NSScrollView();scroll.hasVerticalScroller=true;scroll.borderType = .bezelBorder
  journal.isEditable=false;journal.font = .monospacedSystemFont(ofSize:11,weight:.regular);journal.autoresizingMask=[.width]
  scroll.documentView=journal;view.addArrangedSubview(scroll)
  for v in [picker,note,metric,chartScroll,scroll] as [NSView] {v.widthAnchor.constraint(equalTo:view.widthAnchor,constant:-40).isActive=true}
  scroll.heightAnchor.constraint(equalToConstant:100).isActive=true
  drawCharts()
 }
 func launch(_ args:[String]) {
  guard task == nil,canStart() else {state.stringValue="Сначала останови текущую операцию";return}
  let p=Process(),pipe=Pipe()
  p.executableURL=URL(fileURLWithPath:"/usr/bin/env")
  p.arguments=["python3",Bundle.main.resourcePath!+"/models.py"]+args
  var env=ProcessInfo.processInfo.environment;env["PATH"]="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin";env["PYTHONUNBUFFERED"]="1";p.environment=env
  p.standardOutput=pipe;p.standardError=pipe;pending=""
  buttons.forEach{$0.isEnabled=false};stop.isEnabled=args[0]=="test"
  pipe.fileHandleForReading.readabilityHandler={h in
   let data=h.availableData
   if data.isEmpty {h.readabilityHandler=nil;return}
   DispatchQueue.main.async {self.consume(String(decoding:data,as:UTF8.self))}
  }
  p.terminationHandler={process in DispatchQueue.main.async {
   self.task=nil;self.buttons.forEach{$0.isEnabled=true};self.stop.isEnabled=false
   if args[0]=="rename" || args[0]=="delete" {self.refreshList()}
  }}
  do {try p.run();task=p} catch {state.stringValue=error.localizedDescription;buttons.forEach{$0.isEnabled=true};stop.isEnabled=false}
 }
 func consume(_ text:String) {
  pending+=text
  while let end=pending.firstIndex(of:"\n") {
   let line=String(pending[..<end]);pending=String(pending[pending.index(after:end)...])
   guard let data=line.data(using:.utf8),let d=try? JSONSerialization.jsonObject(with:data) as? [String:Any] else {continue}
   if let models=d["models"] as? [[String:Any]] {
    picker.removeAllItems();picker.addItems(withTitles:models.compactMap{$0["name"] as? String})
    state.stringValue="Моделей: \(models.count)"
   }
   if let text=d["text"] as? String {state.stringValue=text;journal.textStorage?.append(NSAttributedString(string:text+"\n"));journal.scrollToEndOfDocument(nil)}
   if d["event"] as? String == "result" {results.append(d);drawCharts()}
  }
 }
 @objc func changeMetric(){drawCharts()}
 func drawCharts() {
  bars.arrangedSubviews.forEach{bars.removeArrangedSubview($0);$0.removeFromSuperview()}
  let selected=max(0,metric.selectedSegment)
  let keys=["speed","load","memory"]
  let units=["токенов/с","с","ГБ"]
  let key=keys[selected]
  chartHint.stringValue=selected == 0 ? "Больше — быстрее. Среднее двух запросов." : selected == 1 ? "Меньше — быстрее. Загрузка после выгрузки модели из Ollama." : "Меньше — экономнее. Объём загруженной модели по данным Ollama."
  let sorted=results.sorted {
   let a=$0[key] as? Double ?? (selected == 0 ? -1 : Double.infinity)
   let b=$1[key] as? Double ?? (selected == 0 ? -1 : Double.infinity)
   return selected == 0 ? a>b : a<b
  }
  let maximum=max(0.001,sorted.compactMap{$0[key] as? Double}.max() ?? 1)
  if sorted.isEmpty {
   let empty=NSTextField(wrappingLabelWithString:"Нажми «Тест моделей», чтобы сравнить установленные модели. Результаты появятся по мере завершения тестов.")
   empty.textColor = .secondaryLabelColor
   bars.addArrangedSubview(empty)
   empty.widthAnchor.constraint(equalTo:bars.widthAnchor).isActive=true
  }
  for row in sorted {
   let value=row[key] as? Double
   let label=NSTextField(wrappingLabelWithString:row["name"] as? String ?? "")
   label.font = .systemFont(ofSize:13,weight:.medium)
   label.isSelectable=true
   let number=NSTextField(labelWithString:value.map{String(format:"%.2f %@",$0,units[selected])} ?? "нет данных")
   number.font = .monospacedDigitSystemFont(ofSize:13,weight:.semibold)
   number.setContentCompressionResistancePriority(.required,for:.horizontal)
   let header=NSStackView(views:[label,NSView(),number])
   let bar=ResultBar()
   bar.fraction=(value ?? 0)/maximum
   bar.tint=[NSColor.systemTeal,NSColor.systemBlue,NSColor.systemPurple][selected]
   let card=NSStackView(views:[header,bar])
   card.orientation = .vertical;card.alignment = .leading;card.spacing=7
   card.edgeInsets=NSEdgeInsets(top:8,left:8,bottom:12,right:8)
   bars.addArrangedSubview(card)
   card.widthAnchor.constraint(equalTo:bars.widthAnchor).isActive=true
   header.widthAnchor.constraint(equalTo:card.widthAnchor,constant:-16).isActive=true
   bar.widthAnchor.constraint(equalTo:header.widthAnchor).isActive=true
   bar.heightAnchor.constraint(equalToConstant:12).isActive=true
  }
 }
 @objc func refreshList(){launch(["list"])}
 @objc func testModels(){
  guard task == nil,canStart() else {state.stringValue="Сначала останови текущую операцию";return}
  results=[];drawCharts();journal.string="";launch(["test"])
 }
 @objc func stopTask(){task?.terminate();state.stringValue="Остановлено";stop.isEnabled=false}
 @objc func deleteModel(){
  guard let name=picker.titleOfSelectedItem else{return}
  let a=NSAlert();a.messageText="Удалить \(name)?";a.addButton(withTitle:"Удалить");a.addButton(withTitle:"Отмена")
  if a.runModal() == .alertFirstButtonReturn {launch(["delete",name])}
 }
 @objc func renameModel(){
  guard let name=picker.titleOfSelectedItem else{return}
  let a=NSAlert();a.messageText="Новое имя модели";let field=NSTextField(frame:NSRect(x:0,y:0,width:360,height:24));field.stringValue=name;a.accessoryView=field
  a.addButton(withTitle:"Переименовать");a.addButton(withTitle:"Отмена")
  if a.runModal() == .alertFirstButtonReturn {
   let new=field.stringValue.trimmingCharacters(in:.whitespacesAndNewlines)
   if !new.isEmpty && new != name {launch(["rename",name,new])}
  }
 }
}


class FlippedChartView: NSView {
 override var isFlipped: Bool {true}
}
class ResultBar: NSView {
 var fraction: Double = 0
 var tint=NSColor.systemTeal
 override func draw(_ dirtyRect:NSRect) {
  NSColor.quaternaryLabelColor.setFill()
  NSBezierPath(roundedRect:bounds,xRadius:4,yRadius:4).fill()
  tint.setFill()
  let fill=NSRect(x:0,y:0,width:bounds.width*min(1,max(0,fraction)),height:bounds.height)
  NSBezierPath(roundedRect:fill,xRadius:4,yRadius:4).fill()
 }
}
