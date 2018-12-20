let launchButton = document.getElementById('launcher');
launchButton.addEventListener('click', () => {
  console.log('Launching simulator');
  var sw = window.open('build/index.html', 'simulatorTab', 'width=900,height=600,top=0,left=0,location=no,menubar=no,titlebar=no');
  setTimeout(() => {
    console.log('simulatorWindow', sw);
    window.simulatorWindow = sw;
  }, 500);
});
