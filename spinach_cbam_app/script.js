(function(){
  'use strict';
  function $(id){ return document.getElementById(id); }
  function showError(msg){ console.error(msg); var r=$("result"); if(r) r.innerHTML='<span class="label">Error:</span> '+String(msg); }
  function bytesToSize(bytes){
    if (!bytes && bytes !== 0) return "";
    var sizes = ['B','KB','MB','GB']; var i = 0; var n = bytes;
    while (n >= 1024 && i < sizes.length-1){ n/=1024; i++; }
    return n.toFixed(n<10?1:0) + ' ' + sizes[i];
  }

  function renderProbs(rows){
    if (!rows || !rows.length) return '';
    var head = '<div class="prob-table"><table class="pretty"><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>';
    var body = rows.map(function(r){
      var pct = (r.prob*100).toFixed(2) + '%';
      return '<tr><td>'+r.class+'</td><td>'+pct+'</td></tr>';
    }).join('');
    return head + body + '</tbody></table></div>';
  }

  function checkHealth(){
    fetch("/health").then(function(r){ return r.json(); }).then(function(h){
      var node = $("health");
      if (!node) return;
      node.innerHTML = '<div>'+ h.device +', ' + (h.resolved_arch || 'unknown arch') +'</div>'
        + '<div>IMG_SIZE: '+ h.img_size +'</div>'
        + '<div>Classes: '+ (h.classes||[]).join(", ") +'</div>'
        + (h.missing_keys && h.missing_keys.length ? '<div>Missing keys: '+ h.missing_keys.length +'</div>' : '')
        + (h.unexpected_keys && h.unexpected_keys.length ? '<div>Unexpected keys: '+ h.unexpected_keys.length +'</div>' : '');
    }).catch(function(err){
      showError("Health check failed: " + err.message);
    });
  }

  function main(){
    var imageInput = $("imageInput");
    var fileName = $("fileName");
    var predictBtn = $("predictBtn");
    var previewImg = $("previewImg");
    var resultDiv = $("result");
    var explain = $("explain");
    var tta = $("tta");
    var camMethod = $("camMethod");

    checkHealth();

    imageInput.addEventListener("change", function(){
      var file = imageInput.files && imageInput.files[0];
      if (!file) { fileName.textContent = "No file chosen"; fileName.title = "No file chosen"; return; }
      var nice = file.name + (file.size ? " — " + bytesToSize(file.size) : "");
      fileName.textContent = nice; fileName.title = nice;
      var reader = new FileReader();
      reader.onload = function(e){ previewImg.src = e.target.result; previewImg.style.display = "block"; };
      reader.readAsDataURL(file);
    });

    predictBtn.addEventListener("click", function(){
      var file = imageInput.files && imageInput.files[0];
      if (!file){ resultDiv.innerHTML = '<span class="label">Pick an image first.</span>'; return; }
      predictBtn.disabled = true;
      resultDiv.textContent = "Running inference…";
      var fd = new FormData();
      fd.append("file", file);
      fd.append("explain", (explain && explain.checked) ? "true" : "false");
      fd.append("tta", (tta && tta.checked) ? "true" : "false");
      fd.append("cam_method", (camMethod && camMethod.value) ? camMethod.value : "layercam");
      fetch("/predict", { method:"POST", body:fd }).then(function(resp){
        return resp.json().then(function(data){ return {ok:resp.ok, data}; });
      }).then(function(res){
        if (!res.ok) throw new Error(res.data && res.data.error || "Predict failed");
        var d = res.data || res;
        var prob = d.probability || 0;
        var predPct = (prob * 100).toFixed(2);
        var html = ''
          + '<div><span class="label">Prediction:</span> ' + d.label + ' ('+ predPct +'%)'
          + (d.tta_used ? ' — with TTA' : '') + '</div>';
        if (d.inference_time_ms){ html += '<div><span class="label">Inference time:</span> ' + d.inference_time_ms + ' ms</div>'; }
        if (d.cam_method){ html += '<div><span class="label">Explainer:</span> ' + d.cam_method + '</div>'; }
        if (d.probs){ html += '<h4>Class Probabilities</h4>' + renderProbs(d.probs); }
        if (d.heatmap){ html += '<h4>Heatmap</h4><div class="heatmap"><img src="'+ d.heatmap +'" alt="Heatmap"/></div>'; }
        resultDiv.innerHTML = html;
      }).catch(function(err){
        showError(err.message || err);
      }).finally(function(){
        predictBtn.disabled = false;
      });
    });
  }

  if (document.readyState === "complete" || "interactive" === document.readyState){ setTimeout(main,0); }
  else { document.addEventListener("DOMContentLoaded", main); }
})();

(function(){ try{ var y = document.getElementById('year'); if (y) y.textContent = new Date().getFullYear(); }catch(e){} })();
