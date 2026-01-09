import React, { useState, useCallback } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

export function useToast() {
  const [toasts, setToasts] = useState([])
  const push = useCallback((msg, type='success') => {
    const id = Math.random().toString(36).slice(2)
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3000)
  }, [])
  return { toasts, push }
}

export function ToastContainer({ toasts }) {
  return (
    <div className="fixed top-4 right-4 space-y-2 z-50">
      <AnimatePresence>
        {toasts.map(t => (
          <motion.div key={t.id} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className={`px-3 py-2 rounded shadow ${t.type==='error' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
            {t.msg}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}