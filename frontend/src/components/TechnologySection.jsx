import React from 'react'
import { motion } from 'framer-motion'

export default function TechnologySection() {
    const techs = [
        {
            title: "Neural Radiance Fields (NeRF)",
            desc: "Utilizes implicit neural representations to synthesize novel views from sparse 2D inputs.",
            icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="1.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
            )
        },
        {
            title: "Vision Transformers",
            desc: "Captures long-range dependencies in the image to understand global geometry structure.",
            icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            )
        },
        {
            title: "Differentiable Rendering",
            desc: "Enables end-to-end optimization of the 3D mesh parameters directly from image pixels.",
            icon: (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
            )
        }
    ]

    return (
        <section className="py-24 relative overflow-hidden">
            <div className="absolute inset-0 bg-brand-darker/50"></div>
            <div className="max-w-7xl mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    <span className="text-brand-accent font-mono text-sm tracking-widest uppercase">Under the Hood</span>
                    <h2 className="text-4xl font-display font-bold text-white mt-2">Powered by Advanced AI</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {techs.map((t, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.1 }}
                            className="bg-surface-glass backdrop-blur-md p-8 rounded-2xl border border-white/5 hover:border-brand-primary/30 transition-colors group"
                        >
                            <div className="w-16 h-16 bg-brand-primary/10 rounded-xl flex items-center justify-center text-brand-primary mb-6 group-hover:scale-110 transition-transform">
                                {t.icon}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-3">{t.title}</h3>
                            <p className="text-gray-400 leading-relaxed">{t.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    )
}
