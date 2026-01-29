const o=function(e,t){navigator.clipboard.writeText(e).then(()=>(t.show=!0,setTimeout(()=>{t.show=!1},5e3),!0)).catch(console.warn)};export{o as c};
